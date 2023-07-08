# routing

# Milestone 1 - Understand the problem. Understand how the problem will be solved.

This project sets to solve a route-planning algorithm in public transit for optimizing the most convenient route to use by suggesting the appropriate Matatu terminal or stage. This is achieved by taking the current location and final destination coordinates of the user to solve the shortest path problem algorithm. While the inputted coordinates may not be part of the route nodes, it is imperative to solve for the most convenient terminal by considering the weight of the trips and distance between the current location and the pick-up terminal or stage as well as the distance between the final destination and drop-off terminal. The weight of the trip could be the distance, cost, time and capacity of a specific route.

# Milestone 1 – Data. Explain the Source and attributes

 This project is carried out using GTFS static data, which is the most common standardized format for public transit schedules and geographic information. GTFS was created in 2005 for transit agencies to describe their schedules, trips, routes, and stops data in an open-source format that can be used for Google Transit Web-based trip planner. The GTFS .zip file downloaded from Digital Matatu website contains 10 text files that form a database-like structure with every file as one table, and can import these CSV files directly into our program to run queries on them.
In routes.txt file, a path on which public transport vehicles travel are defined on a route_short_name and route_long_name column using a route_id. A route is provided by a public transport agency (defined in agencies.txt), and can be serviced once or more times in a day. Every trip on a route is defined with a route_id, service_id, trip_id, trip_headsign, direction_id, and shape_id columns respectively in the trips.txt file. While the shape.txt and frequencies.txt are optional requirement, it provides the rules for mapping vehicle travel paths.
 The stops made during a trip are also defined as stop times in stop_times.txt. A stop time does not contain information about the stop itself: it only links a trip to a stop, and includes some additional information such as the time of arrival and departure. Stops itself are defined in the stops.txt file, including information such as their name, location coordinates and entrances.
 calendar.txt, calender_dates.txt and feed_info.txt are conditionally required meaning the field or file is required under certain conditions like exceptions for the services defined in the calender.txt, which is the service dates specified using a weekly schedule with start and end dates. If calendar.txt is omitted, then calendar_dates.txt is required and must contain all dates of service. 

# Milestone 2 – Machine Learning to be used.

When finding the most convenient bus stops based on the current location of the user, I first searched for the “k” most nearest neighboring stops in the dataset to the current location (k = no. of stops). Likewise on the destination side, we use the same K Nearest Neighbor algorithm to get the closest bus stops to the user’s destination while including their differences in walking distance.
In this algorithm, finding closest similar points is a core-deciding factor on where the user will either depart or arrive at by finding the great-circle distance between two points on a sphere given their longitudes and latitudes using the Haversine formula.   
This is achievable by reading the stops latitude and stops longitude columns stored inside the stop.txt file and taking input for current location and destination in coordinates format. Meanwhile carrying the stop id column for referencing later as shown below where k = 5.

#extract needed files
sh_df = pd.read_csv(zf.open('stops.txt'))
df = sh_df[['stop_id','stop_lat','stop_lon']]
#Setup Balltree using df as reference dataset
# Use Haversine formula to calculate distance between points on the earth from lat/long
tree = BallTree(np.deg2rad(df[['stop_lat','stop_lon']].values), metric='haversine')
# Setup distance queries (points for which we want to find nearest neighbors)
#location points in coordinates
other_data = """stop_id stop_lat stop_lon
new -1.28797555000001 36.88334477"""
# Use StringIO to allow reading of string as CSV
df_other = pd.read_csv(StringIO(other_data), sep = ' ')
query_lats = df_other['stop_lat']
query_lons = df_other['stop_lon']
# Find closest stops in reference dataset for each in df_other
# use k = 5 for 5 closest stops
distances, indices = tree.query(np.deg2rad(np.c_[query_lats1, query_lons1]), k = 5)
r_km = 6371000 # multiplier to convert to meters (from unit distance)
stp_df=[]
dis_df=[]
for stop_id, d, ind in zip(df_other1['stop_id'], distances, indices):
    for i, index in enumerate(ind):
        stp_id = (df['stop_id'][index])
        stp_df.append(stp_id)
        dis = f"{d[i]*r_km:.4f}"
        dis_df.append(dis)
        stp_dis = list(zip(stp_df,dis_df))
	#stops id and distance from df_other (cods 4 either dest or depart)
        stp_dis_df = pd.DataFrame (stp_dis, columns=['stop_id', 'distance'])

With this information, we can therefore run another query on the stop_times.txt to associate each stop_id with its corresponding trip_id and stop_sequence. 


# Algorithm Procedure:
# Step One
i.	Import necessary libraries for the program. i.e. pandas, numpy, BallTree from sklearn.neighbor, zipfile and  math 
ii.	Read current location and destination location in coordinates, store them in a dataframe (df_other)
iii.	Read stop.txt, stop_times.txt, routes.txt, trips.txt, shapes.txt, frequency.txt file and get stop_id, stop_lat, stop_lon columns only from stop.txt as df.
iv.	Using Haversine formula to calculate the distance in radius between each point in df and the two df_other from departure and destination.
v.	Using BallTree to query for k = 70 closest stops in coordinates that are near the current location and same for the destination. Store the new data in dataframe stp_dis_df with stop_id and distance columns.
vi.	Use stp_dis_df data frames to query with stop_times.txt for their trip_id and stop_sequence. Store the new df in trips_dfc after sorting in ascending order using distance and stop_sequence as merits and dropping of duplicate trip_id’s but keep first in order.
vii.	Merge the two trips_dfc (from departure and destination) with their corresponding trip_id, while dropping objects without a match and sorting in ascending order. We’ll have a new df with ‘trip_id’, ‘stop_id_x’,  ‘distance_x’, ‘stop_id_y ’, ‘distance_y’ : drops all stop_sequence_ columns, stop_id_x is the departure point while stop_id_y for destination point.
viii.	Add columns distance_y and distance_x to get total walking distance. Get the minimum row based on t_distance column in a data frame named trps1.
ix.	If no data was found on trps1, then repeat step one (v) – (viii) with k increased to 2150.
x.	Check if total distance is greater than 1.5km and distance_y and distance_x is also greater than 1km, also if total distance is greater than 1.5km and distance_y is greater than 1km and also if total distance is greater than 1.5km and distance_x is greater than 1km.
xi.	Create a function to repeat step one (iv) – (v) with coordinates in df_other being CBD Nairobi. 
Step Two
i.	if total distance is greater than 1.5km and distance_y and distance_x is also greater than 1km,
ii.	compare the dataframe obtained in step one (x)  with step one (vi) individually repeating step one (vii) – (vii) . resulting with two different trips
N/B stop_id_x will always reps the stop for departure while stop_id_y will rep stop for destination.
iii.	Use trps1t and trp1w to map two route connection
Step Three
i.	if total distance is greater than 1.5km and distance_y is greater than 1km only
ii.	Use stop_id_y from trps1 in step one (viii) as our new departure point to the destination, extract the coordinates from stop.txt using the stop_id as save as new df_other (departure) and destination being the original point.
iii.	Repeat step one (v)-(viii). Get the resulting new trp5. 
iv.	Use trps1 and trp5 to map two route

Step Four
i.	if total distance is greater than 1.5km and distance_x is greater than 1km only
ii.	Use stop_id_x from trps1 in step one (viii) as our new departure point to the original departure, extract the coordinates from stop.txt using the stop_id as save as new df_other (destination) and departure being the original point.
iii.	Repeat step one (v)-(viii). Get the resulting new trp6. 
iv.	Use trps1 and trp6 to map two route
Step Five 
i.	If conditions in steps two, three and four and not satisfied then use trps1 from step one to map a single route


# Milestone 3 – Tools used

Pandas – Makes it simple to do many of the program’s tasks associated with working on data analysis and manipulations, this includes:
1.	Loading and saving data: Since all the GTFS data are stored in .txt files, pandas allows us to load the files in a 2 Dimension that is rows and columns and save it for analysis later.
import pandas as pd
sh_df = pd.read_csv(zf.open('stops.txt'))
stop_times = pd.read_csv(zf.open('stop_times.txt'))
shapes = pd.read_csv(zf.open('shapes.txt'))
trips = pd.read_csv(zf.open('trips.txt'))
routes = pd.read_csv(zf.open('routes.txt'))



2.	Data visualization: consists of transforming  columns to a standard scale
      df = sh_df[['stop_id','stop_lat','stop_lon']]    #get only 3 columns from stop.txt
3.	Merges and joins: merge more or less does the same thing as join. Both methods are used to combine two DataFrames together, though merge requires specifying the columns as a merge key.
#cross check for trip_id using stop_id from stop_times.txt
trips_df = stop_time_df[stop_time_df.set_index(['stop_id']).index.isin(stp_dis_df1.set_index(['stop_id']).index)]
trips_dfc = pd.merge(trips_df, stp_dis_df1, on=['stop_id'], how='left')

4.	Statistical analysis: provides tools for reading and writing data into data structures and files. Like sorting and dropping, unordered and duplicate data respectively.
trips_dfcd = trips_dfc.sort_values(by=['distance'], ascending=True).drop_duplicates(subset='trip_id', keep='first')
5.	Data Inspection: Panda also helps in viewing data for verification and debugging purposes, before, during, or after a translation. 
trps = trps[['trip_id', 'stop_id_x', 'distance_x', 'stop_id_y', 'distance_y', 't_distance']]
trps1 = trps[trps['t_distance'] == trps['t_distance'].min()]
totalDis = trps1['t_distance']

Numpy - with it’s large collection of high-level mathematical functions to operate on arrays, numpy.deg2rad() is mathematical function that helps us to convert angles from degrees to radians.
import numpy as np
tree = BallTree(np.deg2rad(df[['stop_lat','stop_lon']].values), metric='haversine')
Scikit-Learn (Sklearn) – is a machine learning library that supports supervised learning algorithms like the nearest neighbor based classification type. Where in the algorithm classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point. While Balltree, divides points based on radial distances to a center. Nearest neighbor techniques is considered more efficient for fast-generalized N-point problems with a time complexity of O(N*log(N)).
from sklearn.neighbors import BallTree
#Setup Balltree using df as reference dataset
# Use Haversine formula to calculate distance between points on the earth from lat/lo
tree = BallTree(np.deg2rad(df[['stop_lat','stop_lon']].values), metric='haversine')
# Find closest stops in reference dataset for each in df_other
# use k = 70 for 70 closest stops since their exists 136 routes in the dataset
distances, indices = tree.query(np.deg2rad(np.c_[query_lats1, query_lons1]), k)


# Milestone 5 – Data Processing

Collection: 

Working on the route planner algorithm, it needs all necessary bus stops associated with a trip or route in the region of study. The data is collected from the Digital Matatu website (http://digitalmatatus.com/map.html) that has details of all bus stops and routes in the region of Nairobi stored in a GTFS format.

Preparation : 
The collected data is in a raw gtfs format which is directly read into the program. Since there several txt files zipped together, will need zipfile module and pandas libraries to open the zip file and read their contents in the program for execution. i.e.
# Create DataFrame from gtfs_feed
# Create DataFrame from gtfs_feed
#unzipping the file
zf = zipfile.ZipFile('GTFS_FEED_2019.zip')
#extract needed files
sh_df = pd.read_csv(zf.open('stops.txt'))

Input: The input data can be in the form that may not be machine-readable, so to convert the input data to the readable form we use StringIO modules to allow reading of stings inputs into as csv.
Output: In this stage, we use tkinker map view to procured results from the algorithm that can be easily inferred by the user.
# Missing Values:  
From a validation report generated using the Canonical GTFS Schedule validator , we find out stop_times.txt file is missing a time_points  and shape_dist_traveled columns. Even though they are all optionally required, time_points column indicates if arrival and departure times for a stop are strictly adhered to by the vehicle or if they are instead approximate and/or interpolated times. This field allows a GTFS producer to provide interpolated stop-times, while indicating that the times are approximate. Shape_dist_traveled would have provided the actual distance traveled along the associated shape, from the first stop to the last stop specified in this record
The data also is missing fare_attribute.txt and fare_rules.txt files. Although they are also optional files, the fare_rules.txt table specifies how fares in fare_attribute.txt apply to an itinerary. That is; Fare depending on origin or destination stations, Fare depending on which zones the itinerary passes through and Fare depending on which route the itinerary uses.
# Inaccurate Values: 
Due to luck of time_point column in stop_times.txt file as discussed above both arrival time and departure times are rendered inaccurate and unfit for training the algorithm.
Other Inaccuracy occurs on feed_info.txt on feed_start_date and feed_end_date columns having expired. Although they are optionally required, the suggested expiration date from the Canonical GTFS Schedule validator is 2023/01/01 the feed end date is 2022/12/31.

# Curse of Dimentionality:
K-Nearest Neighbor algorithms perform better with a lower number of features than a large number of features. We can say that when the number of features increases then it requires more data. Increase in dimension also leads to the problem of overfitting. To avoid overfitting, the needed data will need to grow exponentially as it increases the number of dimensions. To avoid the problem of the curse of dimensionality, i performed feature selection approach on the data using exhaustive feature selection techniques.
Before implementing the techniques, note that noisy and less important data like agency.txt, calender.txt, calender_dates.txt, feed_info.txt and frequency.txt were not necessary to include in the model.
By trying and making each possible combination like enquiring for a single connecting route, it is possible to pick the best sets. Since there are 136 routes recorded on the dataset, implementing step one (i-Viii) with K = 70;
i)	 will get all closest 70 stops from both current location and destination in distance,
ii)	 Merge them in ascending order, 
iii)	Drop duplicate trips, while keeping the first elements on the data frame  
iv)	Finally, find the one with a minimum total walking distance added from both ends.
Works best for a person that needs to board one matatu and use only one route to reach their destination.
Example in the Code (current location is Umoja estate and destination is near Achieves center in town):
df = sh_df[['stop_id','stop_lat','stop_lon']]
temp_df = df
stop_time_df = stop_times[['trip_id','stop_id','stop_sequence']]
#Setup Balltree using df as reference dataset
# Use Haversine formula to calculate distance between points on the earth from lat/long
tree = BallTree(np.deg2rad(df[['stop_lat','stop_lon']].values), metric='haversine')

def closestKnn(df_other1, query_lats1, query_lons1,k):
    # Find closest stops in reference dataset for each in df_other
    # use k = 70 for 70 closest stops since their exists 136 routes in the dataset
    distances, indices = tree.query(np.deg2rad(np.c_[query_lats1, query_lons1]), k)
    r_km = 6371000 # multiplier to convert to meters (from unit distance)
    stp_df=[]
    dis_df=[]
    for stop_id, d, ind in zip(df_other1['stop_id'], distances, indices):
        for i, index in enumerate(ind):
            stp_id = (df['stop_id'][index])
            stp_df.append(stp_id)
            dis = f"{d[i] * r_km:.4f}"
            dis_df.append(dis)
            stp_dis = list(zip(stp_df, dis_df))
            stp_dis_df = pd.DataFrame(stp_dis, columns=['stop_id', 'distance']) #gets the stop_id and respective distances
    return stp_dis_df
pass
# Setup distance queries (points for which we want to find nearest neighbors)
#location points in coordinates
other_data = """stop_id stop_lat stop_lon
new -1.2858143069958488 36.89552028737499"""
# Use StringIO to allow reading of string as CSV
df_other1 = pd.read_csv(StringIO(other_data), sep = ' ')
query_lats1 = df_other1['stop_lat']
query_lons1 = df_other1['stop_lon']

dest_data = """stop_id stop_lat stop_lon
dest -1.2855300409249406 36.826278115418816"""  # destination points
df_other2 = pd.read_csv(StringIO(dest_data), sep = ' ')
query_lats2 = df_other2['stop_lat']
query_lons2 = df_other2['stop_lon']

stp_dis_df2=closestKnn(df_other1,query_lats1,query_lons1, k=70)
trips_df1= stop_time_df[stop_time_df.set_index(['stop_id']).index.isin(stp_dis_df2.set_index(['stop_id']).index)]
trips_dfc1 = pd.merge(trips_df1, stp_dis_df2, on=['stop_id'], how='left')
trips_dfc1 = trips_dfc1.sort_values(by=['distance'], ascending=True)

stp_dis_df1=closestKnn(df_other1=df_other2,query_lats1=query_lats2, query_lons1=query_lons2,k=70)
#cross check for trip_id using stop_id from stop_times.txt
trips_df = stop_time_df[stop_time_df.set_index(['stop_id']).index.isin(stp_dis_df1.set_index(['stop_id']).index)]
trips_dfc = pd.merge(trips_df, stp_dis_df1, on=['stop_id'], how='left')
trips_dfcd = trips_dfc.sort_values(by=['distance'], ascending=True).drop_duplicates(subset='trip_id', keep='first')
trps = trips_dfc1.merge(trips_dfcd, on='trip_id', how='inner').sort_values(by=['distance_y'], ascending=True)
trps['t_distance'] = trps['distance_x'].astype(float) + trps['distance_y'].astype(float)
trps = trps[['trip_id', 'stop_id_x', 'distance_x', 'stop_id_y', 'distance_y', 't_distance']]
trps1 = trps[trps['t_distance'] == trps['t_distance'].min()]
totalDis = trps1['t_distance']
temp_trps1 = trps1
totalDis = totalDis.values[0]
w_distance_x = trps1['distance_x'].astype(float)
w_distance_x = w_distance_x.values[0]
w_distance_y = trps1['distance_y'].astype(float)
w_distance_y = w_distance_y.values[0]
stop_id_y = trps1['stop_id_y']
stop_id_y = stop_id_y.values[0]
stop_id_x = trps1['stop_id_x']
stop_id_x = stop_id_x.values[0]
if stop_id_y==stop_id_x:
    print('trip is too close, just walk\n',trps1)
else:
        depart_pt = trps1.iat[0,1]
        stage1 = sh_df[sh_df.values==depart_pt]
        sg1 = stage1.iat[0, 1]
        arrival_pt = trps1.iat[0, 3]
        stage2 = sh_df[sh_df.values == arrival_pt]
        sg2 = stage2.iat[0, 1]
        trip_id = trps1.iat[0, 0]
        trp_rt = trips[trips.values==trip_id]
        rt_id = trp_rt.iat[0, 0]
        rt = routes[routes.values==rt_id]
        rt_name = rt.iat[0, 3]
        mat_no = rt.iat[0, 2]
        print('Walk for ',w_distance_x,'mtrs',' to ',sg1,'stage,\n',
              'Board Bus No',mat_no,'using',rt_name, 'route,\n','Your next drop off will be',sg2,'stage,',
              '\ntotal walking distance from current location to destination is\n', totalDis, 'meters.\n',trps1)

But what if a person is required to use two or three connecting routes to reach their destination from the current location. The above code let’s say finds a matching trip id under the same condition, it will prompt the user to use the single route though having to cover a great distance.
 
 The model might also not get matching trips and produce an error.
Trying to avoid missing a trip in the model we repeat Step One (i-viii) using k=2150 since they’re 4284 stops in stop.txt files to be analyzed, and induce three if else loop conditions to handle the problem of a great walking distance; That is in Step two, three and four.
# Milestone 5 - Training 
Since it is not ideal to use the training dataset for testing the model accuracy and effectiveness while avoiding overfitting and underfitting. Therefore, split up “df” dataset into inputs (X) and target (y). Our input will be every column except ‘trip_id’ because ‘trip_id’ is what we will be attempting to predict. Therefore, ‘trip_id’ will be our target.
Scikit-learn has a ‘train_test_split’ function from model selection that makes it posibble to split our dataset into training and testing data. Set ‘test_size’ to 0.2 so that 20% of all the data will be used for testing, which leaves 80% of the data as training data for the model to learn from. create a new k-NN classifier model, set ‘n_neighbors’ to 70, algorithm to BallTree and Haversine formula for metrics. Then, using the ‘knn.fit’ function to pass in our training data as parameters to fit our model to the training data.
from sklearn.model_selection import train_test_split,
from sklearn.neighbors import KNeighborsClassifier,
import numpy as np
zf = zipfile.ZipFile('GTFS_FEED_2019.zip')
#extract needed files
sh_df = pd.read_csv(zf.open('stops.txt'))
df=df[['trip_id', 'stop_lat','stop_lon','stop_sequence']]
y = df.stop_id
X = df.drop('stop_id', axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 70,metric='haversine',algorithm='ball_tree')
# Fit the classifier to the data
knn.fit(X,y)
print('X_train shape:',X_train.shape,'\n','y_train shape:',y_train.shape,'\n','X-test:',X_test.shape,'\n','y_test shape:',y_test.shape,'\n','X shape:',X.shape,'\n','y shape:',y.shape,'\n','data shape:',df.shape)

Printing the X_train.shape, y_train.shape, X_test.shape, y_test.shape, X.shape, y.shape, df.shape we get the results below respectively.
The model might also not get matching trips and produce an error.
Trying to avoid missing a trip in the model we repeat Step One (i-viii) using k=2150 since they’re 4284 stops in stop.txt files to be analyzed, and induce three if else loop conditions to handle the problem of a great walking distance; That is in Step two, three and four.
