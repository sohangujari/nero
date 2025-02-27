import React from 'react'

function Footer() {
  return (
    <div className='flex flex-col'>
        <div className='flex justify-between items-center px-16 py-6 bg-white'>
            <div className='flex flex-col'>
                <span className='font-bold'>Nero AI</span>
                <span className='w-4/8 mt-2 text-gray-600 font-light'>Nero is a AI-Powered chatbot app that allows users to have conversationwith virtual assistant.</span>
            </div>
            <div className='flex gap-24'>
                <div className='flex flex-col'>
                    <span className='font-bold'>Company</span>
                    <span className='mt-2 text-gray-600 font-light'>About</span>
                    <span className='mt-2 text-gray-600 font-light'>Careers</span>
                </div>
                <div className='flex flex-col'>
                    <span className='font-bold'>Help</span>
                    <span className='mt-2 text-gray-600 font-light'>FAQs</span>
                    <span className='mt-2 text-gray-600 font-light'>Contact Support</span>
                </div>
                <div className='flex flex-col'>
                    <span className='font-bold'>Access</span>
                    <span className='mt-2 text-gray-600 font-light'>Login</span>
                    <span className='mt-2 text-gray-600 font-light'>Request Demo</span>
                </div>
            </div>
        </div>
        <hr className='w-11/12 mx-auto text-gray-200'/>
        <span className='px-16 py-6 text-gray-600 font-light'>Copyright &#169; 2025. All rights reserved</span>
    </div>
  )
}

export default Footer