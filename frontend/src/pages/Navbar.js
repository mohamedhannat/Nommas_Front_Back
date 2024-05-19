import React, { useState, useRef, useEffect } from 'react';
import { HiOutlineX, HiMenu } from 'react-icons/hi';
import { getAuth, signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';

const Navbar = ({ setIsAuthenticated, setSelectedLinks, sidebarLinks, setSelectedMenu }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const linkContainerRef = useRef(null);
  const linksRef = useRef(null);
  const navigate = useNavigate();
  const auth = getAuth();

  useEffect(() => {
    if (isMenuOpen) {
      linkContainerRef.current.style.height =
        linksRef.current.getBoundingClientRect().height + 'px';
    } else {
      linkContainerRef.current.style.height = '0px';
    }
  }, [isMenuOpen]);

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      setIsAuthenticated(false);
      navigate('/signin', { replace: true });
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleMenuClick = (menu) => {
    setSelectedLinks(sidebarLinks[menu]);
    setSelectedMenu(sidebarLinks[menu][0].action);  // Set the first link as selected
    setIsMenuOpen(false);
  };

  return (
    <div className="relative bg-white">
      <div className="mx-auto max-w-7xl">
        <div className="relative hidden px-4 py-5 md:block">
          <nav className="flex items-center justify-between">
            <div className="logo">
              <div className="flex items-center px-5 text-2xl font-bold text-gray-600">
                <span className="px-2 text-indigo-600">Nommas</span>
              </div>
            </div>
            <div className="flex items-center capitalize">
              <button
                onClick={() => handleMenuClick('home')}
                className="px-5 py-4 font-medium text-gray-500 capitalize rounded-sm hover:text-gray-900 hover:bg-gray-50"
              >
                Home
              </button>
              <button
                onClick={() => handleMenuClick('settings')}
                className="px-5 py-4 font-medium text-gray-500 capitalize rounded-sm hover:text-gray-900 hover:bg-gray-50"
              >
                Settings
              </button>
              <button
                onClick={() => handleMenuClick('model')}
                className="px-5 py-4 font-medium text-gray-500 capitalize rounded-sm hover:text-gray-900 hover:bg-gray-50"
              >
                Model
              </button>
              <button
                onClick={handleSignOut}
                className="px-5 py-2 ml-4 text-white bg-red-600 rounded-lg hover:bg-red-700"
              >
                Sign Out
              </button>
            </div>
          </nav>
        </div>
        <div className={`absolute py-5 px-4 md:hidden top-0 w-full bg-white overflow-hidden`}>
          <nav className="flex flex-col my-2">
            <div className="flex justify-between">
              <div className="flex items-center px-5 text-2xl font-bold text-gray-600">
                Nommas
                <span className="px-2 text-indigo-600">Nommas</span>
              </div>
              {isMenuOpen ? (
                <button
                  onClick={() => setIsMenuOpen(!isMenuOpen)}
                  className="flex items-center justify-center w-10 h-10 text-gray-500 rounded-sm opacity-100 hover:text-gray-700 hover:bg-gray-100 active:bg-gray-200 focus:outline-none focus:ring focus:ring-indigo-500"
                >
                  <HiOutlineX className="w-6 h-6" />
                </button>
              ) : (
                <button
                  onClick={() => setIsMenuOpen(!isMenuOpen)}
                  className="flex items-center justify-center w-10 h-10 text-gray-500 rounded-sm hover:text-gray-700 hover:bg-gray-100 active:bg-gray-200 focus:outline-none focus:ring focus:ring-indigo-500"
                >
                  <HiMenu className="w-6 h-6" />
                </button>
              )}
            </div>

            <div className={`transition-all ease-linear duration-200 ${isMenuOpen ? 'visible' : 'invisible'}`} ref={linkContainerRef}>
              <div className="flex flex-col" ref={linksRef}>
                <button
                  onClick={() => handleMenuClick('home')}
                  className="px-5 py-4 font-medium text-gray-500 capitalize rounded-sm hover:text-gray-900 hover:bg-gray-50"
                >
                  Home
                </button>
                <button
                  onClick={() => handleMenuClick('settings')}
                  className="px-5 py-4 font-medium text-gray-500 capitalize rounded-sm hover:text-gray-900 hover:bg-gray-50"
                >
                  Settings
                </button>
                <button
                  onClick={() => handleMenuClick('model')}
                  className="px-5 py-4 font-medium text-gray-500 capitalize rounded-sm hover:text-gray-900 hover:bg-gray-50"
                >
                  Model
                </button>
                <button
                  onClick={handleSignOut}
                  className="px-5 py-2 mt-4 text-white bg-red-600 rounded-lg hover:bg-red-700"
                >
                  Sign Out
                </button>
              </div>
            </div>
          </nav>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
