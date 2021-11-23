import pyrebase

config = {
  "apiKey": "AIzaSyCc9R34Svpox90nF6p74eKHCoVQl5I_7eQ",
  "authDomain": "mychatapp-91e9c.firebaseapp.com",
  "databaseURL": "https://mychatapp-91e9c-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "mychatapp-91e9c",
  "storageBucket": "mychatapp-91e9c.appspot.com",
  "messagingSenderId": "536107617441",
  "appId": "1:536107617441:web:bd38f4aaa547c031d8b4e0",
  "measurementId": "G-F4KFWF257K"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
data = {"4": ""}

db.push(data)
