import React from 'react'

function Navbar() {
  return (
    <div className='flex justify-between items-center px-16 py-6 bg-white'>
        <span className='font-bold'>Nero AI</span>
        <div className='flex gap-4'>
            <button className='px-4 py-1 border border-gray-200 rounded-full bg-black text-white font-light'>Home</button>
            <button className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light'>Features</button>
            <button className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light'>Pricing</button>
            <button className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light'>Blog</button>
        </div>
        <button className='px-4 py-1 border border-gray-200 rounded-full text-gray-600 font-light'>Try For Free</button>
    </div>
  )
}

export default Navbar