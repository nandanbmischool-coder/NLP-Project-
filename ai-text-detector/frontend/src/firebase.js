import { initializeApp } from "@firebase/app";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from "@firebase/auth";

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};
export const isFirebaseConfigured = Object.values(config).every(Boolean);
const auth = isFirebaseConfigured ? getAuth(initializeApp(config)) : null;
const missingConfigError = new Error("Firebase is not configured. Add the VITE_FIREBASE_* values to frontend/.env.local.");
export const signIn = (email, password) => auth ? signInWithEmailAndPassword(auth, email, password).then(({ user }) => user) : Promise.reject(missingConfigError);
export const signUp = (email, password) => auth ? createUserWithEmailAndPassword(auth, email, password).then(({ user }) => user) : Promise.reject(missingConfigError);
export const signOutUser = () => auth ? signOut(auth) : Promise.resolve();
