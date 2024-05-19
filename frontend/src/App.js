import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Navbar from './pages/Navbar';
import Sidebar from './pages/Sidebar';
import MainArea from './pages/MainArea';
import SignIn from './login/SignIn';
import SignUp from './login/SignUp';
import { AppProvider } from './pages/context'; // Adjust the import path as necessary
import { HiHome, HiTicket } from 'react-icons/hi';
import { MdSettings } from 'react-icons/md';

const sidebarLinks = {
  home: [
    { id: 'home', text: 'Home', icon: <HiHome />, action: 'home' },
  ],
  settings: [
    { id: 'settings', text: 'Settings', icon: <MdSettings />, action: 'settings' },
  ],
  model: [
    { id: 'annotate', text: 'TrainModel', icon: <HiTicket />, action: 'train' },
    { id: 'dataset', text: 'TestModel', icon: <HiTicket />, action: 'test' },
    { id: 'matching', text: 'Matching', icon: <HiTicket />, action: 'match' },
  ],
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [selectedMenu, setSelectedMenu] = useState('home'); // Default active menu
  const [links, setLinks] = useState(sidebarLinks.home);

  return (
    <AppProvider>
      <Router>
        <div className="flex flex-col h-screen">
          {isAuthenticated ? (
            <>
              <Navbar setIsAuthenticated={setIsAuthenticated} setSelectedLinks={setLinks} sidebarLinks={sidebarLinks} setSelectedMenu={setSelectedMenu} />
              <div className="flex">
                <Sidebar selectedMenu={selectedMenu} setSelectedMenu={setSelectedMenu} links={links} />
                <MainArea selectedMenu={selectedMenu} />
              </div>
            </>
          ) : (
            <Routes>
              <Route path="/signin" element={<SignIn setIsAuthenticated={setIsAuthenticated} />} />
              <Route path="/signup" element={<SignUp setIsAuthenticated={setIsAuthenticated} />} />
              <Route path="*" element={<Navigate to="/signin" />} />
            </Routes>
          )}
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
