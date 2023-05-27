import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('Spaceshiptitanic/train.csv')
test = pd.read_csv('Spaceshiptitanic/test.csv')

print("あなたがいる宇宙船は損傷し、異常事態によって別の次元に飛ばされる危険がある。あなたは別の次元に飛ばされてしまうのか？")
HomePlanet = input("出発地点(Earth or Europa or Mars):")
CryoSleep = str(input("航海中に仮死状態に置かれることを選択しますか？(Yes or No):"))
if CryoSleep == "No":
    CryoSleep = False
elif CryoSleep == "Yes":
    CryoSleep = True
else:
    CryoSleep = ""
Age = float(input("年齢:"))
VIP = input("あなたはVIP会員ですか？(Yes or No):")
if VIP == "Yes":
    VIP = True
elif VIP == "No":
    VIP = False
else:
    VIP = ""
Name = str(input("名前:"))


df = pd.DataFrame(columns=['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'],index=[0])
test = test.append({'HomePlanet': HomePlanet, 'CryoSleep': CryoSleep, 'Age': Age, 'VIP': VIP, 'Name': Name, 'PassengerId':'SSSS' }, ignore_index=True)


As = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for a in As:
  test[a] = test[a].fillna(test[a].mean())
  train[a] = train[a].fillna(train[a].mean())
x = train[['HomePlanet','CryoSleep','Age','VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
y = train[['Transported']]
x = pd.get_dummies(x, columns=['HomePlanet', 'CryoSleep', 'VIP'])

x_train, x_test, y_train, y_test = train_test_split(x ,y , random_state=0)
scalar = StandardScaler()
scalar.fit(x_train)
x_train_scaled = scalar.transform(x_train)
x_test_scaled = scalar.transform(x_test)

rf = RandomForestClassifier()
rf.fit(x_train_scaled,y_train)

x_for_submit = test[['HomePlanet','CryoSleep','Age','VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
submit = test[['Name']]

x_for_submit = pd.get_dummies(x_for_submit, columns=['HomePlanet', 'CryoSleep', 'VIP'])
x_for_submit['Age'] = x_for_submit['Age'].fillna(x_for_submit['Age'].mean)

x_for_submit_scaled = scalar.transform(x_for_submit)

submit['Transported'] = rf.predict(x_for_submit_scaled)
if submit.loc[4277,"Transported"] == True:
    print("宇宙船は損傷し、{}はその異常によって別の次元に飛ばされた。".format(submit.loc[4277,"Name"]))
else:
    print("宇宙船は損傷したが、{}はその異常によって別の次元に飛ばされなかった。".format(submit.loc[4277,"Name"]))
