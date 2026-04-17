#### Forest Trend Mapper Application using Geemap, Solara, and Earth Engine, March 2026 ###
### Research Article:https://www.mdpi.com/1999-4907/16/5/777###

# Import necessary libraries
import os
import ee
import geemap
import solara
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import threading
import solara.lab
from ipyleaflet import FullScreenControl

# Initialize Earth Engine
try:
    if "EE_SERVICE_ACCOUNT" in os.environ and "EE_PRIVATE_KEY" in os.environ:
        # Formats the key correctly by replacing escaped newlines
        private_key = os.environ.get("EE_PRIVATE_KEY").replace('\\n', '\n')
        credentials = ee.ServiceAccountCredentials(
            os.environ.get("EE_SERVICE_ACCOUNT"),
            key_data=private_key
        )
        ee.Initialize(credentials)
        print("Earth Engine initialized via Service Account.")
    else:
        ee.Initialize()
except Exception as e:
    print("Fallback to local authentication.")
    ee.Authenticate()
    ee.Initialize()

#Define color palettes and climate variables
TAU_PALETTE = ["#8c2d04", "#cc4c02", "#ec7014", "#fe9929", "#fdd49e", "#d9f0d3", "#addd8e", "#78c679", "#31a354", "#006837"]
SLOPE_PALETTE = ["#67001f", "#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac", "#053061"]
CLIMATE_BANDS = {
    'Summer Max Temp (°C)': 'Temp',
    'Summer Max Temp (°C)(1-yr Lag)': 'Temp_Lag1',
    'Summer Total Precip (mm)': 'Precip',
    'Summer Total Precip (mm)(1-yr Lag)': 'Precip_Lag1',
    'Spring Snow Water Equiv (mm)': 'SWE',
    'Summer Soil Moisture (mm)': 'Soil'
}

#Reactive state
class State:
    start_year = solara.reactive(2000)
    end_year = solara.reactive(2024)
    map_index = solara.reactive("NDVI")
    reducer = solara.reactive("median")
    apply_sig_mask = solara.reactive(True)
    show_mtbs = solara.reactive(True)
    show_ak_fire = solara.reactive(True)
    climate_var = solara.reactive("Summer Max Temp (°C)")
    fire_info = solara.reactive(None)

    dark_mode = solara.reactive(False)
    last_point = solara.reactive(None)
    annual_col = solara.reactive(None)
    is_processing = solara.reactive(False)
    point_df = solara.reactive(None)
    is_fetching_point = solara.reactive(False)
    error_msg = solara.reactive(None)

    tau_img = solara.reactive(None)
    slope_img = solara.reactive(None)
    sig_mask = solara.reactive(None)
    current_roi = solara.reactive(None)
    
    download_urls = solara.reactive({})
    is_generating_urls = solara.reactive(False)
    show_info_modal = solara.reactive(False)
    show_map_help = solara.reactive(False)


#GEE functions

def apply_scale_factors(img):
    optical = img.select('SR_B.*').multiply(0.0000275).add(-0.2)
    return img.addBands(optical, None, True)

def fmask(img):
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    snow = qa.bitwiseAnd(1 << 5).neq(0)
    water = qa.bitwiseAnd(1 << 7).neq(0)
    
    mask = cloud.Or(shadow).Or(snow).Or(water).Not()
    return img.updateMask(mask)

def add_all_indices(img):
    ndvi = img.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    nbr = img.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
    ndmi = img.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')
    return img.addBands([ndvi, nbr, ndmi])

def prep_oli(img):
    img = apply_scale_factors(img)
    renamed = img.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL']
    )
    renamed = fmask(renamed)
    return add_all_indices(renamed).toFloat()

def prep_etm(img):
    img = apply_scale_factors(img)
    renamed = img.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL']
    )
    renamed = fmask(renamed)
    return add_all_indices(renamed).toFloat()

def get_landsat_jja(roi, start_year, end_year):
    start = ee.Date.fromYMD(start_year, 1, 1)
    end = ee.Date.fromYMD(end_year, 12, 31)
    
   
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start, end).map(prep_oli)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start, end).map(prep_oli)
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(roi).filterDate(start, end).map(prep_etm)
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(roi).filterDate(start, end).map(prep_etm)
    
    return l9.merge(l8).merge(l7).merge(l5).filter(ee.Filter.calendarRange(6, 8, 'month'))

def get_annual_composites(col, roi, start_year, end_year, reducer_type):
    years = ee.List.sequence(start_year, end_year)
    
    # Climate Data including 1-year lag
    climate_col = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterBounds(roi) \
        .filterDate(ee.Date.fromYMD(start_year - 1, 1, 1), ee.Date.fromYMD(end_year, 12, 31))

    def create_annual(y):
        y = ee.Number(y)
        y_prev = y.subtract(1)

        # Landsat Summer Composite
        yc = col.filter(ee.Filter.calendarRange(y, y, 'year')).select(['NDVI', 'NBR', 'NDMI'])
        empty = ee.Image([0, 0, 0]).rename(['NDVI', 'NBR', 'NDMI']).updateMask(ee.Image(0))
        
        # Handling the reducer type (median vs max)
        veg_comp = ee.Image(ee.Algorithms.If(
            yc.size().gt(0), 
            ee.Image(ee.Algorithms.If(reducer_type == 'median', yc.median(), 
                 ee.Algorithms.If(reducer_type == 'mean', yc.mean(), yc.max()))),
            empty
        )).rename(['NDVI', 'NBR', 'NDMI']).clip(roi)

        # Climate Bands
        y_clim = climate_col.filter(ee.Filter.calendarRange(y, y, 'year'))
        temp = y_clim.filter(ee.Filter.calendarRange(6, 8, 'month')).select('tmmx').mean().multiply(0.1).rename('Temp')
        precip = y_clim.filter(ee.Filter.calendarRange(6, 8, 'month')).select('pr').sum().rename('Precip')
        swe = y_clim.filter(ee.Filter.calendarRange(3, 5, 'month')).select('swe').mean().rename('SWE')
        soil = y_clim.filter(ee.Filter.calendarRange(6, 8, 'month')).select('soil').mean().multiply(0.1).rename('Soil')
        
        # Lagged Climate
        y_clim_lag = climate_col.filter(ee.Filter.calendarRange(y_prev, y_prev, 'year'))
        temp_lag = y_clim_lag.filter(ee.Filter.calendarRange(6, 8, 'month')).select('tmmx').mean().multiply(0.1).rename('Temp_Lag1')
        precip_lag = y_clim_lag.filter(ee.Filter.calendarRange(6, 8, 'month')).select('pr').sum().rename('Precip_Lag1')

        clim_comp = ee.Image([temp, precip, swe, soil, temp_lag, precip_lag]).clip(roi)
        year_band = ee.Image.constant(y).rename('year').toFloat()
        
        return veg_comp.addBands(clim_comp).addBands(year_band) \
            .set({'year': y, 'system:time_start': ee.Date.fromYMD(y, 7, 15).millis()}) \
            .toFloat()

    return ee.ImageCollection.fromImages(years.map(create_annual))

def fetch_point_data(coords):
    
    State.last_point.value = coords
    State.is_fetching_point.value = True
    State.point_df.value = None
    State.fire_info.value = None

    def worker():
        try:
            if State.annual_col.value is None:
                return
            
            # Leaflet clicks return [lat, lon], EE expects [lon, lat]
            geom = ee.Geometry.Point(coords[::-1])

            try:
                fires = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')
                intersecting = fires.filterBounds(geom).getInfo().get('features', [])
                
                if intersecting:
                    fire_names = []
                    for f in intersecting:
                        props = f.get('properties', {})
                        # MTBS uses 'Incid_Name' for the fire name and 'Ig_Date' for the ignition date
                        name = props.get('Incid_Name', 'Unknown').title()
                        date = props.get('Ig_Date')
                        if date:
                            year = str(pd.to_datetime(date, unit='ms').year)
                        else:
                            year = "Unknown Year"

                        fire_names.append(f"{name} Fire ({year})")
                    
                    # Join them together in case overlapping fires exist at this point
                    State.fire_info.value = " | ".join(fire_names)
            except Exception as e:
                print(f"Fire check failed: {e}")
            
            # Request all relevant bands in a single call
            bands_to_fetch = ['year', 'NDVI', 'NBR', 'NDMI'] + list(CLIMATE_BANDS.values())
            data = State.annual_col.value.select(bands_to_fetch).getRegion(geom, 30).getInfo()
            
            # Convert to DataFrame and drop the EE metadata columns
            df = pd.DataFrame(data[1:], columns=data[0])
            
            # Ensure our columns are strictly numeric for Plotly
            for col in bands_to_fetch:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Push the completed dataframe to state
            State.point_df.value = df
            
        except Exception as e:
            print(f"Error fetching point data: {e}")
        finally:
            State.is_fetching_point.value = False

    # Start the background thread so the Solara UI doesn't freeze
    threading.Thread(target=worker).start()

# the function that connects gee code to UI components
def run_analysis(m):
    # Clear any previous errors
    State.error_msg.value = None
    
    # Get the ROI from the map component
    roi = m.user_roi
    if roi is None:
        State.error_msg.value = "No ROI found. Please use the draw tools on the map to draw a polygon first."
        return

    # Set loading state to trigger the spinner
    State.is_processing.value = True
    
    # Create a background worker so the Solara UI doesn't freeze while Earth Engine processes the data
    def worker():
        try:
            col = get_landsat_jja(roi, State.start_year.value, State.end_year.value)
            annual_col = get_annual_composites(col, roi, State.start_year.value, State.end_year.value, State.reducer.value)
            State.annual_col.value = annual_col

            # Trend and Slope Calculation
            map_idx = State.map_index.value
            xy = annual_col.select(['year', map_idx])
            
            tau = xy.reduce(ee.Reducer.kendallsCorrelation()).select(f"{map_idx}_tau").rename('tau')
            slope = xy.reduce(ee.Reducer.sensSlope()).select('slope').rename('slope')
            
            # Significance Masking
            n_img = annual_col.select(map_idx).count()
            z_score = tau.multiply(3).multiply(n_img.multiply(n_img.subtract(1)).sqrt()) \
                          .divide(n_img.multiply(2).add(5).multiply(2).sqrt()).abs()
            
            sig_mask = z_score.gte(1.96)

            # Update State
            State.tau_img.value = tau
            State.slope_img.value = slope
            State.sig_mask.value = sig_mask
            State.current_roi.value = roi
            State.download_urls.value = {}

            # Update Map Layers safely
            layers_to_remove = ['Kendall τ (All)', 'Sen Slope (All)', 'Significant Tau', 'Historical Fires']
            
            # Safely remove old analysis layers
            for layer in list(m.layers):
                if getattr(layer, 'name', '') in layers_to_remove:
                    m.remove_layer(layer)
                    
            # Clear old colorbars
            try:
                m.remove_colorbars()
            except AttributeError:
                pass 
            
            # Set up correct visual parameters dictionaries
            tau_vis = {'min': -1, 'max': 1, 'palette': TAU_PALETTE}
            slope_vis = {'min': -0.01, 'max': 0.01, 'palette': SLOPE_PALETTE}
            
            # Add new layers
            show_base = not State.apply_sig_mask.value
            m.add_layer(tau.clip(roi), tau_vis, 'Kendall τ (All)', show_base)
            m.add_layer(slope.clip(roi), slope_vis, 'Sen Slope (All)', False)
            
            if State.apply_sig_mask.value:
                m.add_layer(tau.updateMask(sig_mask).clip(roi), tau_vis, 'Significant Tau', True)
                m.add_layer(slope.updateMask(sig_mask).clip(roi), slope_vis, 'Significant Slope', False)
            
            # Use the safe vis_params syntax for colorbars
            m.add_colorbar(vis_params=tau_vis, label="Kendall τ", layer_name="Tau Legend", position="bottomleft")
            m.add_colorbar(vis_params=slope_vis, label="Sen Slope", layer_name="Slope Legend", position="bottomleft")
            
            if State.show_mtbs.value:
                mtbs = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1').filterBounds(roi)
                m.add_layer(mtbs, {'color': 'darkgray'}, 'MTBS Fire Perimeters')

            if State.show_ak_fire.value:
                ak_fire = ee.FeatureCollection("projects/ee-ssahoo2/assets/AK_fire_history").filterBounds(roi)
                m.add_layer(ak_fire, {'color': "gray"}, 'AK Fire History (Asset)') # Dark Orange
            m.centerObject(roi, 10)
            
        except Exception as e:
            # PUSH the error directly to the web page so we can see it!
            State.error_msg.value = f"Earth Engine Error: {str(e)}"
            print(f"Terminal log: {e}")

        finally:
            # Turn off the loading spinner when everything is finished or failed
            State.is_processing.value = False

    # Start the worker thread
    threading.Thread(target=worker).start()

#UI components
@solara.component
def TrendInfoModal():
    # Only renders the dialog if the reactive state is True
    if State.show_info_modal.value:
        with solara.v.Dialog(
            v_model=True, 
            on_v_model=lambda x: State.show_info_modal.set(False), 
            width="900px",
            persistent=False
        ):
            with solara.v.Card(style={"padding": "45px","z-index":"9999"}):
                
                solara.Markdown(r'''
                This application uses non-parametric statistics to detect vegetation changes, ensuring results are robust against outliers and sensor noise.
                
                ### 1. Kendall’s Tau ($\tau$)
                Measures the ordinal association between year and vegetation index. A positive $\tau$ indicates an increasing trend (greening), while a negative $\tau$ indicates a decreasing trend (browning).

                **Calculation:** It is a non-parametric measure of rank correlation. It counts the number of "concordant" (increasing) vs. "discordant" (decreasing) pairs of observations over time.
                ### 2. Theil-Sen Slope
                Calculates the median rate of change per year. It is significantly more robust to "bad years" or extreme events than standard linear regression.
                
                **Calculation:** This calculates the median of all slopes between all possible pairs of points in the time series.
                ### 3. Significance Masking
                Uses a Z-score calculation to determine if the trend is statistically significant at the selected p-value ($p \leq 0.05$).

                ### 4. Vegetation Indices
                - **NDVI (Normalized Difference Vegetation Index):** Measures vegetation vigor and density.
                NDVI = (NIR − RED)/(NIR + RED)
                - **NBR (Normalized Burn Ratio):** Assesses burn severity and regrowth.
                NBR = (NIR − SWIR)/(NIR + SWIR)
                - **NDMI (Normalized Difference Moisture Index):** Evaluates moisture content in vegetation.
                NDMI = (NIR − MIR)/(NIR + MIR)
                
                ### Fire History Datasets
                * **MTBS (National):** Monitoring Trends in Burn Severity provides perimeters for large fires (>1000 acres in the West) across the US.
                * **AK Fire History:** A specialized regional dataset for Alaska that includes smaller local burn perimeters often missed by national monitoring programs.

                ''')

                solara.HTML(tag="hr", style="margin: 20px 0; border-top: 1px solid #ddd;")
                
                with solara.Row(justify="end"):
                    solara.Button("Got it", on_click=lambda: State.show_info_modal.set(False), color="primary")
@solara.component
def TrendMapperUI(m):
    with solara.Card(
        title="", # We'll build a custom title row below
        style={"min-height": "400px", "margin": "10px"}
    ):
        # Header Row with Title and Info Icon
        with solara.Row(justify="space-between", style={"align-items": "center", "margin-bottom": "20px"}):
            solara.Text("Map Trends", style={"font-size": "24px", "font-weight": "bold", "color": "#2fa4da"})
            solara.Button(
                icon_name="mdi-information-outline", 
                on_click=lambda: State.show_info_modal.set(True),
                text=True, # Makes it look like just an icon
                color="primary"
            )
    
        solara.Markdown("Draw an ROI and select your parameters below:")

        with solara.Row():
            solara.InputInt("Start Year",value = State.start_year)
            solara.InputInt("End Year",value = State.end_year)

        with solara.Row():
            solara.Select("Map Index", value = State.map_index, values = ["NDVI", "NBR", "NDMI"])
            solara.Select("Composite", value = State.reducer, values = ["mean", "median", "max"])

        
        solara.Checkbox(label="Apply Significance Masking (p ≤ 0.05)", value=State.apply_sig_mask)
        
        with solara.Row(style={"align-items": "center", "margin-bottom": "15px"}):
            solara.Text("Fire Overlays:", style={"font-weight": "regular", "margin-right": "15px"})
            solara.Checkbox(label="MTBS (USA)", value=State.show_mtbs)
            solara.Checkbox(label="AK History (Alaska)", value=State.show_ak_fire)

        # Show the error message if one exists
        if State.error_msg.value:
            solara.Error(State.error_msg.value)

        # Disable button while processing to prevent duplicate EE requests
        solara.Button(
            "Calculate Trends",
            color = "success",
            on_click=lambda: run_analysis(m),
            loading=State.is_processing.value,
            disabled=State.is_processing.value 
        )

@solara.component
def TimeSeriesChart():
    # Theme-aware background
    bg_color = "#1e1e1e" if State.dark_mode.value else "#ffffff"
    template = "plotly_dark" if State.dark_mode.value else "plotly_white"

    with solara.Card(style={"background-color": bg_color, "margin-bottom": "20px"}):
        solara.Text("Time-series Plots", style={"font-size": "24px", "font-weight": "bold", "color": "#2fa4da"})
        if State.is_fetching_point.value:
            return solara.Info("Fetching time series data from Earth Engine...", icon="mdi-cloud-download")
        
        df = State.point_df.value
        if df is None or df.empty:
            return solara.Info("Click a point on the map after running analysis to view time-series plots.")
    
        if State.fire_info.value:
            solara.Markdown(f"**Location History:** {State.fire_info.value}", style={"color": "#ff5252", "font-weight": "bold"})
        
        fig = px.line(df, x="year", y=["NDVI", "NBR", "NDMI"], title="Spectral Trends", template=template)
        
        # Make the chart background transparent to show the Card's color
        fig.update_layout(
            autosize = True,
            font=dict(family="Roboto, Helvetica, Arial, sans-serif", size=12),
            xaxis = dict(title = "Year",showgrid=False),
            yaxis = dict(title = "Index Value",showgrid=True),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=70, r=20, t=40, b=80),
            height=350,
            legend=dict(
                orientation="h",        
                yanchor="top",
                y=-0.4,                
                xanchor="center",
                x=0.5,
                title=None              
            ),
        )
        solara.FigurePlotly(fig,dependencies=[fig])

        solara.HTML(tag="hr", style="margin: 20px 0; border: 0; border-top: 1px solid #ddd;")

            # Climate Correlation Section
        solara.Markdown("### Climate Correlation")

        with solara.Row():
            solara.Select("Vegetation Index", value=State.map_index, values=["NDVI", "NBR", "NDMI"])
            solara.Select("Climate Variable", value=State.climate_var, values=list(CLIMATE_BANDS.keys()))

        c_band = CLIMATE_BANDS[State.climate_var.value]
        v_idx = State.map_index.value
        
        if c_band not in df.columns or v_idx not in df.columns:
            return solara.Error("Selected data not available at this point.")

        fig_clim = go.Figure()
        # Using professional green (#4CAF50) and red (#F44336)
        fig_clim.add_trace(go.Scatter(x=df['year'], y=df[v_idx], name=v_idx, line=dict(color='#4CAF50', width=2.5)))
        fig_clim.add_trace(go.Scatter(x=df['year'], y=df[c_band], name="Climate Variable", yaxis="y2", line=dict(color='#F44336', dash='dot')))

        fig_clim.update_layout(
            autosize = True,
            title=f"{v_idx} vs {State.climate_var.value}",
            font=dict(family="Roboto, Helvetica, Arial, sans-serif", size=12),
            xaxis =dict(showgrid=False),
            yaxis=dict(title=v_idx, title_font=dict(color="#4CAF50"), tickfont=dict(color="#4CAF50"),showgrid=True),
            yaxis2=dict(title=State.climate_var.value, overlaying="y", side="right", title_font=dict(color="#F44336"), tickfont=dict(color="#F44336"),showgrid=True, tickmode="sync"),
            legend=dict(
                orientation="h",        
                yanchor="top",
                y=-0.35,               
                xanchor="center",
                x=0.5
            ),
            template=template,
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=70, r=70, t=50, b=80)
        )
        solara.FigurePlotly(fig_clim,dependencies=[fig_clim])
    
        solara.HTML(tag="hr", style="margin: 20px 0; border: 0; border-top: 1px solid #ddd;")

        DataExport()
        solara.HTML(tag="hr", style="margin: 20px 0; border: 0; border-top: 1px solid #ddd;")
                
        solara.Text("Data Sources", style={"font-size": "18px", "font-weight": "bold", "margin-bottom": "10px", "display": "block"})
        solara.Markdown(r'''
        * **Climate Data:** [TerraClimate: Monthly Climate and Climatic Water Balance](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE) 
        * **US Fire Perimeters:** [Monitoring Trends in Burn Severity (MTBS)](https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_MTBS_burned_area_boundaries_v1)
        * **Alaska Fire History:** [AICC Fire History Shapefiles](https://fire.ak.blm.gov/predsvcs/maps.php)
        ''')
        
@solara.component
def MapDownloader():
    # Only show if an analysis has actually been run
    if State.tau_img.value is None or State.current_roi.value is None:
        return solara.Text("")

    def generate_links():
        State.is_generating_urls.value = True
        
        def worker():
            try:
                roi = State.current_roi.value
                tau = State.tau_img.value
                slope = State.slope_img.value
                mask = State.sig_mask.value
                
                # Apply the significance mask
                sig_tau = tau.updateMask(mask)
                sig_slope = slope.updateMask(mask)
                
                urls = {}
                
                # Helper to fetch URL
                def get_url(img):
                    return img.getDownloadURL({
                        'region': roi,
                        'scale': 30, # Landsat native resolution
                        'format': 'GEO_TIFF',
                        'maxPixels': 1e10 # Prevent total failure on slightly large maps
                    })

                # Fetch URLs from Earth Engine
                urls['Trend Map (Kendall τ)'] = get_url(tau)
                urls['Slope Map (Sen Slope)'] = get_url(slope)
                if State.apply_sig_mask.value:
                    urls['Significant Trend Map'] = get_url(sig_tau)
                    urls['Significant Slope Map'] = get_url(sig_slope)
                
                State.download_urls.value = urls
            except Exception as e:
                print(f"URL generation failed: {e}")
                State.error_msg.value = "Failed to generate maps. The ROI might be too large for direct download."
            finally:
                State.is_generating_urls.value = False

        # Run in background to keep UI responsive
        threading.Thread(target=worker).start()

    with solara.Card(style={"margin": "10px"}):
        solara.Text("Export Map Layers", style={"font-size": "24px", "font-weight": "bold", "color": "#2fa4da"})
        if not State.download_urls.value:
            solara.Markdown("Generate GeoTIFFs clipped to your ROI for use in QGIS/ArcGIS.")
            solara.Button(
                "Generate Download Links", 
                color="primary", 
                on_click=generate_links, 
                loading=State.is_generating_urls.value, 
                icon_name="mdi-link"
            )
            solara.Info("Note: Direct downloads fail if the region is too massive (>32MB). Keep ROIs reasonably sized.")
        else:
            solara.Success("Files ready! Click below to download:")
            # Display the generated URLs as clickable links
            for name, url in State.download_urls.value.items():
                if url:
                    solara.Markdown(f"[{name}]({url})")

@solara.component
def DataExport():
    df = State.point_df.value
    
    # Only show the download button if we actually have data
    if df is not None and not df.empty:
        # Convert the pandas dataframe to a CSV string
        csv_data = df.to_csv(index=False)
        
        with solara.Row(style={"align-items": "center", "justify-content": "space-between"}):
            solara.Text("Download point-specific time series data:", style={"font-size": "14px"})
            
            solara.FileDownload(
                data=csv_data,
                filename="climate_veg_timeseries.csv",
                label="Download CSV",
                icon_name="mdi-download"
            )

@solara.component
def MapHelpModal():
    if State.show_map_help.value:
        with solara.v.Dialog(
            v_model=True, 
            on_v_model=lambda x: State.show_map_help.set(False), 
            width="650px"
        ):
            with solara.v.Card(style={"padding": "45px","z-index": "9999"}):
                solara.Text("How to use Map Tools", style={"font-size": "20px", "font-weight": "bold", "color": "#e4be36"})
                solara.Text("Use the toolbars on the map to select your Region of Interest (ROI) and control the view.", style={"margin-top": "10px", "margin-bottom": "20px", "display": "block"})
                instructions = [
                    ("mdi-pentagon-outline", "**Draw a Polygon:** Click to drop points around your target area. To finish the shape, click on your very first point to close it."),
                    ("mdi-crop-square", "**Draw a Rectangle:** Click and drag to create a box."),
                    ("mdi-pencil", "**Edit a Shape:** Click this to drag the edges or corners of an existing shape."),
                    ("mdi-trash-can-outline", "**Delete a Shape:** Click the trash can, then click on the shape to remove it. Click **Save** next to the toolbar to confirm."),
                    ("mdi-eraser", "**Clear All:** Use the eraser tool in the toolbar to wipe all drawn shapes from the map."),
                    ("mdi-layers-outline", "**Layer Control:** Hover over this icon (top right) to turn specific map layers on and off."),
                    ("mdi-fullscreen", "**Fullscreen:** Click this to expand the map to fill your screen. Press 'Esc' to exit.")
                ]

                with solara.Column(style={"gap": "15px"}):
                    for icon, text in instructions:
                        with solara.Row(style={"align-items": "start"}):
                            # flex-shrink: 0 prevents the icon from squishing if the text wraps to 2 lines
                            solara.v.Icon(
                                children=[icon], 
                                style_="color: #2fa4da; margin-right: 15px; flex-shrink: 0; margin-top: 2px; font-size: 24px;"
                            )
                            solara.Markdown(text)
                solara.HTML(tag="hr", style="margin: 20px 0; border-top: 1px solid #ddd;")
                
                with solara.Row(justify="end"):
                    solara.Button("Got it", on_click=lambda: State.show_map_help.set(False), color="primary")
@solara.component
def Page():

    solara.Style("""
        .leaflet-container {
            z-index: 1 !important;
        }
        /* Prevents Hugging Face iframes from hiding legend text */
        .leaflet-control {
            max-width: none !important;
            overflow: visible !important;
        }
        .widget-html, .widget-html-content {
            overflow: visible !important;
        }
    """)
    # This hidden tool syncs the entire app's CSS (Sidebar, Cards, etc.)
    def sync_theme():
            solara.lab.theme.dark = State.dark_mode.value
            
    solara.use_effect(sync_theme, [State.dark_mode.value])
    
    TrendInfoModal()
    MapHelpModal()

    
    with solara.AppBarTitle():
        solara.Text(" ")
    with solara.AppBar():
        with solara.v.Html(tag="div", style_="display: flex; width: 100%; align-items: center; justify-content: space-between;"):
            
            # Column A: Left Spacer (flex: 1 ensures it takes exactly 1/3 of the space to balance the right side)
            solara.v.Html(tag="div", style_="flex: 1;")
            
            # Column B: Centered Title
            with solara.v.Html(tag="div", style_="flex: 1; text-align: center;"):
                solara.Text("Forest Trend Mapper", style={"font-size": "1.5rem", "font-weight": "bold", "white-space": "nowrap"})
            
            # Column C: Right-Aligned Switch
            with solara.v.Html(tag="div", style_="flex: 1; display: flex; justify-content: flex-end; align-items: center; padding-right: 15px; margin-top: 20px;"):
                solara.Switch(label="Dark Mode", value=State.dark_mode)

    solara.Title("Forest Trend Mapper")
 

    # Map Initialization (Memoized to prevent flickering)
    def init_map():
        m = geemap.Map(
            center=[64.2008, -149.4937],
            zoom=5,
            draw_ctrl=True,
            toolbar_ctrl=False
        )
        m.layout.height = "650px"
        m.add_control(FullScreenControl())
        m.lite_mode = True
        return m

    m = solara.use_memo(init_map, [])

    # Map Logic Side-Effect (Syncs Basemap and Click Listeners)
    def setup_map():
        basemap = "CartoDB.DarkMatter" if State.dark_mode.value else "ESRI.WorldStreetMap"
        m.add_basemap(basemap)

        def handle_click(**k):
            if k.get('type') == 'click':
                fetch_point_data(k.get('coordinates'))
        
        m.on_interaction(handle_click)
        return lambda: m.on_interaction(handle_click, remove=True)

    solara.use_effect(setup_map, [m, State.dark_mode.value])


    # Main Dashboard Content
    with solara.v.Html(tag="div", style_="margin-top: -60px;"):
        with solara.Columns([1,2,1.5]):
            with solara.Column():
            
                TrendMapperUI(m)
                MapDownloader()
                # Funding & Acknowledgments Card
                with solara.Card("Funding & Acknowledgments", style={"margin-top": "0px"}):
                    solara.Markdown(r'''
                    This research was funded by the **USDA National Institute of Food and Agriculture**, McIntire Stennis project [Accession Number: 1026801]. 
                    
                    Additional support was received from the **Troth Yeddha' University of Alaska Fairbanks** PhD Fellowship.
                    ''')
            with solara.Column():
                with solara.Card(style={"margin-top": "10px", "margin-bottom": "10px"}):
                    
                    # Map Header
                    with solara.Row(justify="space-between", style={"align-items": "center", "margin-bottom": "5px"}):
                        solara.Text("Display Map", style={"font-size":"24px","font-weight":"bold" ,"color": "#2fa4da"})

                        solara.Button(
                            icon_name="mdi-information-outline", 
                            on_click=lambda: State.show_map_help.set(True),
                            text=True, 
                            color="primary"
                        )
                    
                    # The Map Container
                    with solara.v.Html(tag="div", style_="position: relative; z-index: 1;"):
                        solara.display(m)

                    # A subtle horizontal divider to separate map from text
                    solara.HTML(tag="hr", style="margin: 20px 0 15px 0; border: 0; border-top: 1px solid #444;")

                    # Links & References Section (Zero top gap)
                    solara.Text("Links & References", style={"font-size": "24px", "font-weight": "bold", "margin-bottom": "15px", "display": "block"})
                    
                    solara.Markdown(r'''
                    [Source Code (GitHub)](https://github.com/Sumana18/Forest-trend-mapper) | [Original Paper (MDPI)](https://www.mdpi.com/1999-4907/16/5/777) | [Feedback Form](https://forms.gle/23woKHYHZbKMySNn8)
                    
                    **Citations:**
                    
                    Application:
                                    
                    Journal Publication: Sahoo, S., et al. (2025). Interplay of Topography, Fire History, and Climate on Interior Alaska Boreal Forest Vegetation Dynamics. *Forests*, 16(5), 777.
                    ''')

            
            with solara.VBox():
                TimeSeriesChart()
