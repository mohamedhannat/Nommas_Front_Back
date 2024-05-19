import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const Box = ({ geometry, label, style }) => (
  <div
    style={{
      ...style,
      position: 'absolute',
      left: `${geometry.x}%`,
      top: `${geometry.y}%`,
      height: `${geometry.height}%`,
      width: `${geometry.width}%`,
    }}
  >
    <span
      style={{
        position: 'absolute',
        top: '-20px',
        left: '0',
        color: 'green',
        backgroundColor: 'white',
        padding: '2px',
        fontSize: '12px',
        border: '1px solid green',
      }}
    >
      {label}
    </span>
  </div>
);

const Detection = () => {
  const videoRef = useRef(null);
  const [detectionGeometry, setDetectionGeometry] = useState([]);
  const [params, setParams] = useState({
    weights: 'runs/train/exp3/weights/best.pt',
    source: 0,
    imgsz: [640, 640],
    conf_thres: 0.25,
    iou_thres: 0.45,
    max_det: 1000,
    device: '',
    classes: null,
    agnostic_nms: false,
    line_thickness: 3,
    hide_labels: false,
    hide_conf: false,
    half: false,
    dnn: false,
  });

  useEffect(() => {
    const fetchVideo = () => {
      if (videoRef.current) {
        videoRef.current.src = 'http://localhost:5000/yolo_video_feed';
      }
    };

    const fetchDetections = () => {
      axios.get('http://localhost:5000/detection_data')
        .then(response => {
          setDetectionGeometry(response.data.detectionGeometry);
        })
        .catch(error => {
          console.error("There was an error fetching detection data!", error);
        });
    };

    fetchVideo();
    const interval = setInterval(fetchDetections, 1000); // Fetch detections every second

    return () => clearInterval(interval);
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setParams(prevParams => ({
      ...prevParams,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append('file', file);
      axios.post('http://localhost:5000/upload_model', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }).then(response => {
        setParams(prevParams => ({
          ...prevParams,
          weights: response.data.path,
        }));
      }).catch(error => {
        console.error("There was an error uploading the file!", error);
      });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    axios.post('http://localhost:5000/start_detection', params)
      .then(response => {
        console.log('Detection started', response.data);
        if (videoRef.current) {
          videoRef.current.src = 'http://localhost:5000/yolo_video_feed?' + new Date().getTime(); // Add a timestamp to force refresh
        }
      })
      .catch(error => {
        console.error("There was an error starting the detection!", error);
      });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-100">
      <h1 className="mb-4 text-3xl font-bold">YOLOv5 Real-Time Detection</h1>
      <form className="flex flex-wrap items-center justify-center w-full mb-4 space-x-4" onSubmit={handleSubmit}>
        <div className="flex flex-col items-center">
          <label className="block mb-1">Model File</label>
          <input
            type="file"
            name="weights"
            onChange={handleFileChange}
            className="p-2 border rounded"
          />
        </div>
        <div className="flex flex-col items-center">
          <label className="block mb-1">Confidence Threshold</label>
          <input
            type="number"
            name="conf_thres"
            value={params.conf_thres}
            onChange={handleChange}
            step="0.01"
            className="p-2 border rounded"
          />
        </div>
        <div className="flex flex-col items-center">
          <label className="block mb-1">IOU Threshold</label>
          <input
            type="number"
            name="iou_thres"
            value={params.iou_thres}
            onChange={handleChange}
            step="0.01"
            className="p-2 border rounded"
          />
        </div>
        <div className="flex flex-col items-center">
          <label className="block mb-1">Max Detections</label>
          <input
            type="number"
            name="max_det"
            value={params.max_det}
            onChange={handleChange}
            className="p-2 border rounded"
          />
        </div>
        <div className="flex flex-col items-center">
          <label className="block mb-1">Line Thickness</label>
          <input
            type="number"
            name="line_thickness"
            value={params.line_thickness}
            onChange={handleChange}
            className="p-2 border rounded"
          />
        </div>
        <div className="flex items-center space-x-2">
          <label className="block mb-1">Hide Labels</label>
          <input
            type="checkbox"
            name="hide_labels"
            checked={params.hide_labels}
            onChange={handleChange}
            className="p-2 border rounded"
          />
        </div>
        <div className="flex items-center space-x-2">
          <label className="block mb-1">Hide Confidences</label>
          <input
            type="checkbox"
            name="hide_conf"
            checked={params.hide_conf}
            onChange={handleChange}
            className="p-2 border rounded"
          />
        </div>
        <button type="submit" className="px-4 py-2 text-white bg-blue-500 rounded">Start Detection</button>
      </form>
      <div className="relative" style={{ width: '640px', height: '480px', border: '4px solid gray' }}>
        <img ref={videoRef} alt="Video Stream" className="w-full h-full" />
        {/* {detectionGeometry.map((geometry, index) => (
          <Box
            key={index}
            geometry={geometry}
            label={geometry.label}
            style={{
              border: 'solid 2px green',
            }}
          />
        ))} */}
      </div>
    </div>
  );
};

export default Detection;
