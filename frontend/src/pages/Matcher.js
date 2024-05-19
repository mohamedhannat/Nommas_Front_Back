import axios from 'axios';
import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

function Matcher() {
    const [epochData, setEpochData] = useState({current: 0, total: 2, progress: 0});

    useEffect(() => {
        const socket = io('http://localhost:5000');
        socket.on('training_progress', data => {
            if (data.epoch) {
                setEpochData(data.epoch);
            }
        });
        return () => socket.disconnect();
    }, []);

    const startTraining = () => {
        axios.get('http://localhost:5000/start-training')
            .then(response => console.log('Training started'))
            .catch(error => console.error('Error starting training:', error));
    };

    return (
        <div>
            <h1>YOLOv8 Training Progress</h1>
            <button onClick={startTraining}>Start Training</button>
            <p>Epoch: {epochData.current} / {epochData.total}, Progress: {epochData.progress.toFixed(2)}%</p>
        </div>
    );
}

export default Matcher;
