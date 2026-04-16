#### Forest Trend Mapper Application using Geemap, Solara, and Earth Engine, March 2026 ###
### Research Article:https://www.mdpi.com/1999-4907/16/5/777###

# Import necessary libraries
import ee
import geemap
import solara
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
from solara.lab import theme as theme

# Initialize Earth Engine
try:
    ee.Initialize(project='forest-trend-testing')
    print("Earth Engine initialized successfully.")
except:
    ee.Authenticate()
    ee.Initialize(project='forest-trend-testing')

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
    p_threshold = solara.reactive(0.05)
    show_fires = solara.reactive(True)
    climate_var = solara.reactive("Summer Max Temp (°C)")
    fire_info = solara.reactive(None)

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
            
            sig_mask = z_score.gte(1.96 if State.p_threshold.value <= 0.05 else 1.64)

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
            m.add_layer(tau.clip(roi), tau_vis, 'Kendall τ (All)', False)
            m.add_layer(slope.clip(roi), slope_vis, 'Sen Slope (All)', False)
            m.add_layer(tau.updateMask(sig_mask).clip(roi), tau_vis, 'Significant Tau', True)
            
            # Use the safe vis_params syntax for colorbars
            m.add_colorbar(vis_params=tau_vis, label="Kendall τ", layer_name="Tau Legend")
            m.add_colorbar(vis_params=slope_vis, label="Sen Slope", layer_name="Slope Legend")
        
            if State.show_fires.value:
                fires = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1').filterBounds(roi)
                m.add_layer(fires, {'color': 'red'}, 'Historical Fires')

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
def TrendMapperUI(m):
    with solara.Card("1. Mapping Trends", style ={"margin":"10px"}):
        solara.Markdown("Draw an ROI and select your parameters below:")

        with solara.Row():
            solara.InputInt("Start Year",value = State.start_year)
            solara.InputInt("End Year",value = State.end_year)

        with solara.Row():
            solara.Select("Map Index", value = State.map_index, values = ["NDVI", "NBR", "NDMI"])
            solara.Select("Composite", value = State.reducer, values = ["mean", "median", "max"])

        solara.InputFloat("Significance (p<=)", value = State.p_threshold)
        solara.Checkbox(label="Overlay Historical Fires", value = State.show_fires)

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
    # Show a loading message while the background thread is running
    if State.is_fetching_point.value:
        return solara.Info("Fetching time series data from Earth Engine...", icon="mdi-cloud-download")
    
    df = State.point_df.value
    if df is None or df.empty:
        return solara.Info("Click a point on the map after running analysis to view trends.")
   
    if State.fire_info.value:
        solara.Markdown(f"**Location History:** {State.fire_info.value}", style={"color": "#d32f2f", "font-size": "1.1em"})
    
    fig = px.line(df, x="year", y=["NDVI", "NBR", "NDMI"], title="Spectral Trends", template="plotly_white")
    fig.update_layout(width=570, height=300)
    solara.FigurePlotly(fig)

@solara.component
def ClimateCorrelation():
    if State.is_fetching_point.value:
        return solara.Info("Fetching climate data...", icon="mdi-cloud-download")
        
    df = State.point_df.value
    if df is None or df.empty:
        return solara.Info("Select a point on the map for Climate Correlation.")

    # Dropdowns are now instantly responsive because they just filter the local DataFrame
    solara.Select("Vegetation Index", value=State.map_index, values=["NDVI", "NBR", "NDMI"])
    solara.Select("Climate Variable", value=State.climate_var, values=list(CLIMATE_BANDS.keys()))

    c_band = CLIMATE_BANDS[State.climate_var.value]
    v_idx = State.map_index.value
    
    if c_band not in df.columns or v_idx not in df.columns:
        return solara.Error("Selected data not available at this point.")

    fig = go.Figure()

    #TODO: I adjusted the plot here to make subplots in an attempt to rectify the issue with the overlaying second y-axis.
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['year'], y=df[v_idx], name=v_idx, line=dict(color='green')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['year'], y=df[c_band], name="Climate", yaxis="y2", line=dict(color='red')), secondary_y=True)

    # Original trace
    # fig.add_trace(go.Scatter(x=df['year'], y=df[v_idx], name=v_idx, line=dict(color='green')))
    # fig.add_trace(go.Scatter(x=df['year'], y=df[c_band], name="Climate", yaxis="y2", line=dict(color='red')))
    
    # I commented out the yaxis information here. If we do not use make_subplots, uncomment these lines
    fig.update_layout(
        title=f"{v_idx} vs {State.climate_var.value}",
        # yaxis=dict(title=v_idx, tickfont=dict(color="green")),
        # yaxis2=dict(title="Climate Value", overlaying="y", 
        #             side="right", tickfont=dict(color="red")),
        template="plotly_white", 
        height=300,
        width=570, 
    )

    #TODO: Title for second y-axis has been commented out for now due to overlapping issue
    fig.update_yaxes(title_text=v_idx, secondary_y=False, tickfont=dict(color="green"), showgrid=False)
    fig.update_yaxes(#title_text="Climate Value", 
        secondary_y=True, tickfont=dict(color="red"), showgrid=False)
    solara.FigurePlotly(fig)

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

    with solara.Card("2. Export Map Layers", style={"margin": "10px"}):
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
        
        with solara.Row(justify="end"):
            solara.FileDownload(
                data=csv_data,
                filename="climate_veg_timeseries.csv",
                label="Download CSV",
                icon_name="mdi-download"
            )
    
@solara.component
def Page():
    # Add theme toggle
    solara.lab.ThemeToggle()
    solara.Title("Forest Trend Mapper")
    
    # Simple bar at the top with the name of the app
    with solara.AppBarTitle():
        solara.Text("Forest Trend Mapper")

    #TODO: Can we center the title for this? Maybe try moving it + the Github/paper information into a 
    # collapsible sidebar instead of below the map?
    # with solara.Sidebar(): 
    #     solara.Markdown(r'''
    #         ### The source code can be found at our [GitHub](https://github.com/Sumana18/Forest-trend-mapper).
    #         ### The original paper can be found [here](https://www.mdpi.com/1999-4907/16/5/777).
    #         #### References                
    #         Sahoo, S., Juday, G. P., Panda, S. K., Genet, H., Brown, D. R. N., & Hutten, K. (2025). Interplay of Topography, Fire History, and Climate on Interior Alaska Boreal Forest Vegetation Dynamics in the 21st Century: A Landsat Time-Series Analysis. Forests, 16(5), 777. https://doi.org/10.3390/f16050777
    #         ''')

    # how the map should be built on startup
    def init_map():
        new_map = geemap.Map(center=[64.2008, -149.4937], zoom=4, height="650px")
        new_map.add_basemap("ESRI.WorldStreetMap") # Added your custom basemap here!

        # Attach the click listener for your point time-series
        new_map.on_interaction(lambda **k: k.get('type')=='click' and fetch_point_data(k.get('coordinates')))
        return new_map

    # use_memo saves the map in memory.
    # This is where 'm' is created so we can pass it to your UI and analysis functions!
    m = solara.use_memo(init_map, [])
    
    # 3. Layout the Sidebar
    # with solara.Sidebar(): 
    #     TrendMapperUI(m) # Passes 'm' to your UI
    #     MapDownloader() 
        
    # 4. Layout the Main Screen (Map on left, Charts on right)
    with solara.Columns([0, 2, 2]):
        with solara.Column():
            TrendMapperUI(m) # Passes 'm' to your UI
            MapDownloader() 
        with solara.Column():
            solara.display(m) # Renders the map we saved in memory

            # Paper and Github information!
            solara.Markdown(r'''
            ### The source code can be found at our [GitHub](https://github.com/).
            ### The original paper can be found [here](https://www.mdpi.com/1999-4907/16/5/777).
            #### References                
            Sahoo, S., Juday, G. P., Panda, S. K., Genet, H., Brown, D. R. N., & Hutten, K. (2025). Interplay of Topography, Fire History, and Climate on Interior Alaska Boreal Forest Vegetation Dynamics in the 21st Century: A Landsat Time-Series Analysis. Forests, 16(5), 777. https://doi.org/10.3390/f16050777
            ''')
            
        with solara.VBox():
            TimeSeriesChart()
            ClimateCorrelation()
            DataExport()
