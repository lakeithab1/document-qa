import requests

def get_current_weather(location):
    url = f'https://wttr.in/{location}?format=j1'

    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(
            f"wwtr.in error: status {response.status_code}"

        )
    try:
        data = response.json()
    except ValueError:
        raise Exception(
            f"Could not find a location named {location}"
        )
    current = data["current_condition"][0]

    today = data["weather"][0] #To select the first forecast "today" from multiple days
#Loop to find the highest chance of rain hourly 
    chance_of_rain = max(
        int(hour["chanceofsnow"])
        for hour in today["hourly"]
    )
#Loop to find the highest chance of snow hourly 
    chance_of_snow = max(
        int(hour["chanceofsnow"])
        for hour in today ["hourly"] 
    )
#Returned dictionary storing all weather info 
    return {
        "location": location, #return the location entered by user
        "temperature_f":float(current["temp_F"]), #Finds the tempeture and convert it into a decimal #
        "feels_like_f": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"].strip(), #remove extra spaces/ select the first desription from list
        "high_f": float(today["maxtempF"]), #Convert the highest temp into a decimal
        "low_f": float(today["mintempF"]), #Convert the lowest temp into a decimal
        "humidity_percent": int(current["humidity"]), #convert text into a whole #
        "wind_speed_mph": float(current["windspeedMiles"]),
        "maximum_chance_of_rain": chance_of_rain, #Add the highest chance of rain 
        "maximum_chance_of_snow": chance_of_snow    } #Adds the highest chance of snow

print(get_current_weather("Syracuse, NY"))
print(get_current_weather("Lima, Peru"))