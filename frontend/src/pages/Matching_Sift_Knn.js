import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const Box = ({ geometry, style }) => (
  <div
    style={{
      ...style,
      position: 'absolute',
      left: `${geometry.x}%`,
      top: `${geometry.y}%`,
      height: `${geometry.height}%`,
      width: `${geometry.width}%`,
    }}
  />
);

const Matching = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [bbox, setBbox] = useState({ x1: 0, y1: 0, x2: 0, y2: 0 });
  const [selectionGeometry, setSelectionGeometry] = useState(null);
  const [detectionGeometry, setDetectionGeometry] = useState(null);
  const [selecting, setSelecting] = useState(false);

  useEffect(() => {
    const fetchVideo = () => {
      if (videoRef.current) {
        videoRef.current.src = 'https://nommas-front-back.onrender.com/video_feed';
      }
    };
    fetchVideo();
  }, []);

  const handleMouseDown = (e) => {
    if (!selecting) return;
    setDrawing(true);
    const rect = canvasRef.current.getBoundingClientRect();
    setBbox({ x1: e.clientX - rect.left, y1: e.clientY - rect.top, x2: 0, y2: 0 });
  };

  const handleMouseMove = (e) => {
    if (!drawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const context = canvasRef.current.getContext('2d');
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    context.strokeStyle = 'red';
    context.lineWidth = 2;
    context.strokeRect(bbox.x1, bbox.y1, e.clientX - rect.left - bbox.x1, e.clientY - rect.top - bbox.y1);
  };

  const handleMouseUp = (e) => {
    if (!selecting) return;
    setDrawing(false);
    const rect = canvasRef.current.getBoundingClientRect();
    const x2 = e.clientX - rect.left;
    const y2 = e.clientY - rect.top;
    setBbox({ ...bbox, x2, y2 });

    const width = ((x2 - bbox.x1) / rect.width) * 100;
    const height = ((y2 - bbox.y1) / rect.height) * 100;
    const x = (bbox.x1 / rect.width) * 100;
    const y = (bbox.y1 / rect.height) * 100;

    setSelectionGeometry({ x, y, width, height });
  };

  const handleSelectClick = () => {
    setSelecting(true);
    setDetectionGeometry(null);
  };

  const handleSaveClick = () => {
    setSelecting(false);
    axios.post('https://nommas-front-back.onrender.com/set_bbox', {
      x1: Math.round(bbox.x1),
      y1: Math.round(bbox.y1),
      x2: Math.round(bbox.x2),
      y2: Math.round(bbox.y2),
    }).then((response) => {
      if (response.data.detectionGeometry) {
        setSelectionGeometry(null); // Clear the drawing bounding box
        setDetectionGeometry(response.data.detectionGeometry);
        // Clear the canvas
        const context = canvasRef.current.getContext('2d');
        context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        // Refresh the video stream
        if (videoRef.current) {
          videoRef.current.src = 'https://nommas-front-back.onrender.com/video_feed?' + new Date().getTime(); // Add a timestamp to force refresh
        }
      }
    }).catch((error) => {
      console.error("There was an error saving the bounding box!", error);
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="mb-4 text-2xl font-bold">Video Stream</h1>
      <div className="flex mb-4 space-x-4">
        <button 
          onClick={handleSelectClick} 
          className={`px-4 py-2 rounded shadow-lg transition duration-300 ${selecting ? 'bg-green-500 text-white' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
        >
          Select BBox
        </button>
        <button 
          onClick={handleSaveClick} 
          className={`px-4 py-2 rounded shadow-lg transition duration-300 ${!selecting ? 'bg-green-500 text-white' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
        >
          Save Selection
        </button>
      </div>
      <div className="relative" style={{ width: '640px', height: '480px', border: '4px solid gray' }}>
        <img ref={videoRef} alt="Video Stream" className="w-full h-full" />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0"
          width="640"
          height="480"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        />
        {detectionGeometry && (
          <Box
            geometry={detectionGeometry}
            style={{
              border: 'solid 2px green',
            }}
          />
        )}
      </div>
    </div>
  );
};

export default Matching;
