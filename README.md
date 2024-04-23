# spaceship-titanic
A Data Wrangling Project based on the famous titanic problem.
Data Source: https://www.kaggle.com/competitions/spaceship-titanic/data
Citation: Addison Howard, Ashley Chow, Ryan Holbrook. (2022). Spaceship Titanic. Kaggle. https://kaggle.com/competitions/spaceship-titanic

Dataset provided by Kaggle contains the following data:
train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
• PassengerId - A unique Id for each passenger. Each Id takes the form "gggg_pp" where "gggg" indicates a group the passenger is traveling with and "pp" is their number within the group. People in a group are often family members, but not always.
• HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
• CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
• Cabin - The cabin number where the passenger is staying. Takes the form "deck/num/side", where "side" can be either "P" for Port or "S" for Starboard.
• Destination - The planet the passenger will be debarking to.
• Age - The age of the passenger.
• VIP - Whether the passenger has paid for special VIP service during the voyage or not.
• RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
• Name - The first and last names of the passenger.
• Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. My task is to predict the value of Transported for the passengers in this set.
sample_submission.csv - A submission file in the correct format.
• PassengerId - Id for each passenger in the test set.
• Transported - The target. For each passenger, predict either "True" or "False"

Using these three datasets, I was to learn more about the relationships between each of the variables and how I can use them to fill in any missing data points as well as predict the outcome of whether the passenger was transported with a certain degree of success. Then I would apply what I learned through the training data on the test data and submit a file matching the format of sample_submission.
