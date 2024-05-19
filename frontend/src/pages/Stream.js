import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

function VideoStream() {
    const [imageSrc, setImageSrc] = useState('');

    useEffect(() => {
        const socket = io('http://localhost:5000/stream');
        socket.on('new_frame', data => {
            const imageBlob = new Blob([data.image], { type: 'image/jpeg' });
            const imageObjectURL = URL.createObjectURL(imageBlob);
            setImageSrc(imageObjectURL);
        });
        return () => {
            socket.disconnect();
        };
    }, []);

    return (
        <div>
            <img src={imageSrc} alt="Video Stream" />
        </div>
    );
}

export default VideoStream;
