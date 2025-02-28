import React from 'react'

import { Link } from "react-router-dom";

function Navbar() {
  return (
    <div className='flex justify-between items-center px-16 py-6 bg-white'>
        <span className='font-bold'>Nero AI</span>
        <div className='flex gap-4'>
            <Link to='/' className='px-4 py-1 border border-gray-200 rounded-full bg-black text-white font-light hover:border-gray-900'>Home</Link>
            <Link to='/' className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light hover:border-gray-900'>Features</Link>
            <Link to='/' className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light hover:border-gray-900'>Pricing</Link>
            <Link to='/blog' className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light hover:border-gray-900'>Blog</Link>
        </div>
        <Link to='/' className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light'>Try For Free</Link>
    </div>
  )
}

export default Navbar