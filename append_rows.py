import csv

rows = [
["wb-01","1899","Darjeeling","Darjeeling","West Bengal",27.0410,88.2663,"High","Rainfall","landslide",
 "1899 Darjeeling landslides referenced in later press coverage as one of the deadliest in regional history, with 72 deaths",
 "The Tribune","https://www.tribuneindia.com/news/national/23-dead-in-darjeeling-as-landslides-rip-through-hills-after-torrential-rain",0],

["wb-02","1950-06-12","Darjeeling","Darjeeling","West Bengal",27.0410,88.2663,"High","Rainfall","landslide",
 "546mm of rainfall in one event triggered landslides across Darjeeling, killing 127 people",
 "The Landslide Blog (AGU)","https://blogs.agu.org/landslideblog/2013/07/31/the-darjeeling-landslide-disaster-of-1950/",0],

["wb-03","1968-10-04","Teesta Bazaar","Kalimpong","West Bengal",26.9975,88.4747,"High","Rainfall","landslide",
 "Four days of continuous rainfall triggered hundreds of landslides across Sikkim and Darjeeling; the Anderson Bridge at Teesta Bazaar was destroyed and around 1000 people were killed regionwide",
 "Wikipedia","https://en.wikipedia.org/wiki/1968_Sikkim_floods",1],

["wb-04","2015-07-01","Mirik","Darjeeling","West Bengal",26.8879,88.1858,"High","Rainfall","landslide",
 "Series of landslides across Darjeeling district killed at least 38 people, with 21 deaths in Mirik alone",
 "The Tribune","https://www.tribuneindia.com/news/archive/nation/38-killed-in-darjeeling-landslides-100967",1],

["wb-05","2015-07-01","Kalimpong","Kalimpong","West Bengal",27.0669,88.4700,"High","Rainfall","landslide",
 "Landslides in Kalimpong-1 and 2, Lava, Sukhia block and Gorubathan killed 17 people after 226mm of rain fell in six hours",
 "Save The Hills","http://savethehills.blogspot.com/2015/07/landslides-in-kalimpong-pictoral-report.html",1],

["wb-06","2025-10-05","Mirik","Darjeeling","West Bengal",26.8879,88.1858,"High","Rainfall","landslide",
 "Torrential rain triggered landslides across the Darjeeling hills; 11 of the reported fatalities occurred in Mirik, the worst-hit area",
 "Business Standard","https://www.business-standard.com/india-news/at-least-23-killed-as-heavy-rain-triggers-landslides-in-darjeeling-hills-125100600004_1.html",1],

["wb-07","2025-10-05","Nagrakata","Jalpaiguri","West Bengal",26.9167,88.9333,"Moderate","Rainfall","landslide",
 "Fatalities from the October 2025 Darjeeling-hills landslide event were also reported from the Nagrakata area of Jalpaiguri district",
 "Business Standard","https://www.business-standard.com/india-news/at-least-23-killed-as-heavy-rain-triggers-landslides-in-darjeeling-hills-125100600004_1.html",1],

["sikkim-17","2025-06-01","Chetan Army Camp","East Sikkim","Sikkim",27.2333,88.5000,"High","Rainfall","landslide",
 "Fatal landslide at the Chetan army camp during extreme early-monsoonal rainfall killed 11 people",
 "ScienceDirect study","https://www.sciencedirect.com/science/article/pii/S2666592125000757",0],

["arunachal-01","2017-07-11","Laptap","Papum Pare","Arunachal Pradesh",27.2500,93.5333,"High","Rainfall","landslide",
 "Massive landslide in Sagalee subdivision destroyed houses in Laptap village, killing 14 people",
 "Hindustan Times","https://www.pressreader.com/india/hindustan-times-patiala/20170712/281603830507035",0],

["arunachal-02","2022-05-16","Itanagar","Papum Pare","Arunachal Pradesh",27.0844,93.6053,"Moderate","Rainfall","landslide",
 "Landslides near Punjabi Dhaba in Itanagar contributed to a death toll that rose to 8 in the state during the event",
 "Millennium Post","https://www.pressreader.com/india/millennium-post/20220521/281809992508826",0],

["arunachal-03","2022-05-16","Koloriang","Kurung Kumey","Arunachal Pradesh",27.9333,93.1500,"Moderate","Rainfall","landslide",
 "Landslides in Kurung Kumey district killed three people during the same May 2022 rainfall event that also struck Itanagar",
 "Millennium Post","https://www.pressreader.com/india/millennium-post/20220517/281943136486332",0],

["nagaland-01","2024-09-04","Pherima","Kohima","Nagaland",25.7500,94.0500,"Moderate","Rainfall","landslide",
 "Mudslides and landslides struck National Highway-29 between Kohima and Dimapur near Pherima and Tseipama, killing 6 people",
 "All India Radio News","https://newsonair.gov.in/6-dead-3-injured-in-nagaland-landslides",1],

["nagaland-02","2026-07-18","Mon","Mon","Nagaland",26.7500,95.0667,"High","Rainfall","landslide",
 "Cloudburst-triggered landslide struck more than 10 locations in Mon district, killing 9 people and injuring 5",
 "Down To Earth","https://www.downtoearth.org.in/climate-change/nine-dead-five-injured-in-landslide-triggered-by-extreme-weather-in-nagaland-say-officials",0],

["nagaland-03","2025-06-01","Kohima","Kohima","Nagaland",25.6751,94.1086,"Moderate","Rainfall","landslide",
 "Nagaland recorded landslide fatalities during the late-May/early-June 2025 monsoon disaster that struck northeast India",
 "Al Jazeera","https://www.aljazeera.com/news/2025/6/1/dozens-killed-in-indias-northeast-after-rains-trigger-floods-landslides",0],

["meghalaya-02","2025-06-01","Shillong","East Khasi Hills","Meghalaya",25.5788,91.8933,"Moderate","Rainfall","landslide",
 "Relentless monsoon rains across northeast India triggered floods and landslides in Meghalaya as part of a regional disaster that killed at least 30 people",
 "Al Jazeera","https://www.aljazeera.com/news/2025/6/1/dozens-killed-in-indias-northeast-after-rains-trigger-floods-landslides",0],

["meghalaya-03","2024-05-28","Shillong","East Khasi Hills","Meghalaya",25.5788,91.8933,"Moderate","Rainfall","landslide",
 "One person was killed and four injured in Meghalaya as Cyclone Remal triggered heavy rain that led to landslides in many places",
 "All India Radio News","https://www.newsonair.gov.in/wp-content/uploads/2024/05/nn-ee-830-1.pdf",0],

["tripura-01","2025-06-01","Kanchanbari","Dhalai","Tripura",23.8333,91.8333,"Moderate","Rainfall","landslide",
 "Tripura confirmed fatalities during the same northeast India monsoon floods and landslides event of late May/early June 2025",
 "Al Jazeera","https://www.aljazeera.com/news/2025/6/1/dozens-killed-in-indias-northeast-after-rains-trigger-floods-landslides",0],

["mizo-02","2024-05-28","Melthum","Aizawl","Mizoram",23.7000,92.6800,"High","Rainfall","landslide",
 "Stone quarry collapse at Melthum on the southern outskirts of Aizawl killed 15 people amid heavy rain from Cyclone Remal",
 "The Tribune","https://www.tribuneindia.com/news/india/death-toll-in-mizoram-landslides-rises-to-29-7-still-missing-625981",0],

["mizo-03","2024-05-28","Hlimen","Aizawl","Mizoram",23.6900,92.6900,"High","Rainfall","landslide",
 "Landslide at Hlimen on the southern edge of Aizawl killed six people during Cyclone Remal-induced heavy rain",
 "Sentinel Assam","https://www.sentinelassam.com/north-east-india-news/mizoram-news/mizoram-tragedy-aizawl-landslide-death-toll-climbs-to-28",0],

["mizo-04","2024-05-28","Hunthar","Aizawl","Mizoram",23.7500,92.6500,"Moderate","Rainfall","landslide",
 "A landslide on National Highway-6 at Hunthar cut off Aizawl from the rest of the country during Cyclone Remal",
 "The Tribune","https://www.tribuneindia.com/news/india/death-toll-in-mizoram-landslides-rises-to-29-7-still-missing-625981",1],

["assam-04","2024-05-28","Kamrup","Kamrup","Assam",26.1500,91.6700,"Moderate","Rainfall","landslide",
 "Three people were killed across Kamrup, Kamrup Metro and Morigaon districts as Cyclone Remal triggered landslides and heavy rain across Assam",
 "All India Radio News","https://www.newsonair.gov.in/death-toll-reaches-23-after-massive-landslide-in-mizoram",0],
]

header = ["id","event_date","locality","district","state","latitude","longitude","severity","trigger","landslide_type","description","source_org","source_url","road_affected"]

with open("historical_landslides.csv", "a", newline="") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow(r)

print("added", len(rows), "rows")
