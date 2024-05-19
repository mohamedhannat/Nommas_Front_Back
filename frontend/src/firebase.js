import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
    apiKey: "AIzaSyAtN0S9mzsYwVvxkagTh5rn9POhvh8PNr4",
    authDomain: "nossam-ab470.firebaseapp.com",
    projectId: "nossam-ab470",
    storageBucket: "nossam-ab470.appspot.com",
    messagingSenderId: "451462609742",
    appId: "1:451462609742:web:5c0990be608111d71afc1f",
    measurementId: "G-866DSE48YK"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export { auth };
