import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyAtN0S9mzsYwVvxkagTh5rn9POhvh8PNr4",
    authDomain: "nossam-ab470.firebaseapp.com",
    projectId: "nossam-ab470",
    storageBucket: "nossam-ab470.appspot.com",
    messagingSenderId: "451462609742",
    appId: "1:451462609742:web:5c0990be608111d71afc1f",
    measurementId: "G-866DSE48YK"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const SignIn = ({ setIsAuthenticated }) => {
    const [email, setEmail] = useState("mohamed.hannat@paintup.fr");
    const [password, setPassword] = useState("Simohannatsam");
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        try {
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            if (userCredential.user.emailVerified) {
                setIsAuthenticated(true);
                navigate("/");
            } else {
                alert("Please verify your email before signing in.");
            }
        } catch (err) {
            console.error("SignIn error:", err.code, err.message); // Log the error code and message
            switch (err.code) {
                case 'auth/invalid-email':
                    setError('The email address is badly formatted.');
                    break;
                case 'auth/user-disabled':
                    setError('This user has been disabled.');
                    break;
                case 'auth/user-not-found':
                    setError('No user found with this email.');
                    break;
                case 'auth/wrong-password':
                    setError('The password is incorrect.');
                    break;
                case 'auth/account-exists-with-different-credential':
                    setError('An account already exists with the same email address but different sign-in credentials.');
                    break;
                case 'auth/operation-not-allowed':
                    setError('Sign-in with email and password is not enabled.');
                    break;
                case 'auth/network-request-failed':
                    setError('Network error. Please check your internet connection and try again.');
                    break;
                case 'auth/too-many-requests':
                    setError('Too many requests. Please try again later.');
                    break;
                case 'auth/internal-error':
                    setError('An internal error occurred. Please try again.');
                    break;
                default:
                    setError('An unexpected error occurred.');
                    break;
            }
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
                <h2 className="mb-6 text-2xl font-bold text-center">Sign In</h2>
                {error && <p className="mb-4 text-red-500">{error}</p>}
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label className="block mb-2 text-sm font-medium text-gray-700">Email</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-3 py-2 text-gray-700 border rounded-lg focus:outline-none focus:border-indigo-500"
                            placeholder="Email"
                        />
                    </div>
                    <div className="mb-6">
                        <label className="block mb-2 text-sm font-medium text-gray-700">Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-3 py-2 text-gray-700 border rounded-lg focus:outline-none focus:border-indigo-500"
                            placeholder="Password"
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full py-2 text-white transition duration-300 bg-indigo-500 rounded-lg hover:bg-indigo-600"
                    >
                        Sign In
                    </button>
                </form>
                <p className="mt-4 text-center">
                    Don't have an account?{" "}
                    <Link to="/signup" className="text-indigo-500 hover:text-indigo-700">
                        Sign Up
                    </Link>
                </p>
            </div>
        </div>
    );
};

export default SignIn;
