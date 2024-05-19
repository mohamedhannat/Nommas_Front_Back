import React from 'react';
import { HiOutlineSearch, HiTicket } from 'react-icons/hi';

const Sidebar = ({ selectedMenu, setSelectedMenu, links }) => {
  return (
    <div className="fixed top-0 left-0 w-64 h-screen min-h-screen px-4 py-8 overflow-y-auto transition-all duration-500 bg-white border-r">
      <h2 className="text-3xl font-semibold text-gray-800">
        <span className="ml-1 text-indigo-500">Nommas</span>
      </h2>
      <div className="relative mt-6">
        <label htmlFor="searchP" className="absolute inset-y-0 left-0 flex items-center pl-3">
          <HiOutlineSearch className="w-5 h-5 text-gray-400 hover:text-gray-500" />
        </label>
        <input
          id="searchP"
          type="text"
          placeholder="Search"
          className="w-full py-3 pl-10 pr-4 text-gray-700 border border-gray-300 rounded-md hover:border-gray-400 focus:outline-none focus:ring focus:border-blue-500"
        />
      </div>
      <nav className="flex flex-col justify-between flex-1 mt-6">
        {links.map((link) => (
          <button
            key={link.id}
            onClick={() => setSelectedMenu(link.action)}
            className={`capitalize flex items-center px-4 py-2 mt-5 text-gray-600 hover:bg-gray-200 hover:text-gray-700 transition-colors duration-200 transform rounded-md ${selectedMenu === link.action ? 'bg-blue-500 text-white' : ''}`}
          >
            {link.icon}
            <span className="mx-4 font-medium">{link.text}</span>
          </button>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;
