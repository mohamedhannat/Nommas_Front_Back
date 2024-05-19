
import React, { useEffect, useState, useRef } from 'react';
// import Annotation from 'react-image-annotation';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import UploadImages from './UploadImages'; // Ensure you have this import

const Annotator = ({ imageFiles, setImageFiles, imagesPreview, setImagesPreview, tags, setTags }) => {
  const [annotations, setAnnotations] = useState([]);
  const [annotation, setAnnotation] = useState({});
  const [selectedImage, setSelectedImage] = useState();
  const undoStackRef = useRef([]);
  const redoStackRef = useRef([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [trainPercent, setTrainPercent] = useState(70);
  const [valPercent, setValPercent] = useState(20);
  const [testPercent, setTestPercent] = useState(10);
  const [datasetFolder, setDatasetFolder] = useState("dataset");
  const [activeTab, setActiveTab] = useState('loaderImage');
  const [showModal, setShowModal] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingCompleted, setTrainingCompleted] = useState(false);
  const [showSaveContinueModal, setShowSaveContinueModal] = useState(false);

  const style = {
    button: "text-white bg-blue-500 py-2 px-4 rounded shadow-lg transition duration-300 hover:bg-blue-600",
    modalButton: "text-white bg-red-500 py-2 px-4 rounded shadow-lg transition duration-300 hover:bg-red-600",
    tabButton: "px-4 py-2 transition duration-300",
    tabButtonActive: "border-b-2 border-blue-500 text-blue-500",
    tabButtonInactive: "text-gray-500 hover:text-blue-500",
    arrowButton: "text-gray-500 hover:text-blue-500 transition duration-300",
    arrowContainer: "flex items-center justify-between mb-4",
    editorSelect: "block w-full p-2 mt-1 border border-gray-300 rounded focus:outline-none",
    undoRedoButton: "bg-blue-500 text-white py-2 px-3 rounded shadow-md transition duration-300 hover:bg-blue-600"
  };

  const colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', '#FF4500', '#DA70D6', '#40E0D0', '#FF69B4', '#BA55D3'];

  useEffect(() => {
    setSelectedImage(imagesPreview[currentIndex]);
  }, [currentIndex, imagesPreview]);

  const handleNextImage = () => {
    if (currentIndex < imagesPreview.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePrevImage = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const getColorForTag = (tag) => {
    return colors[tags.indexOf(tag) % colors.length];
  };

  const Box = ({ children, geometry, style }) => (
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
      {children}
    </div>
  );

  function renderSelector({ annotation }) {
    const { geometry } = annotation;
    if (!geometry) return null;

    return (
      <Box
        geometry={geometry}
        style={{
          border: `solid 2px #E10000`
        }}
      />
    );
  }

  function renderHighlight({ annotation }) {
    if (!annotation.data) return null;

    return (
      <Box
        key={annotation.data.id}
        geometry={annotation.geometry}
        style={{
          border: `solid 3px ${getColorForTag(annotation.data.text)}`,
        }}
      />
    );
  }

  function renderContent({ annotation }) {
    if (!annotation.data) return null;

    return (
      <div
        key={annotation.data.id}
        style={{
          background: getColorForTag(annotation.data.text),
          color: 'white',
          padding: '4px 8px',
          borderRadius: '4px',
          position: 'absolute',
          left: `${annotation.geometry.x}%`,
          top: `${annotation.geometry.y - 4}%`, // Positioned above the bbox
          fontSize: '0.75rem',
          whiteSpace: 'nowrap',
          transform: 'translateY(-100%)', // Ensure it's always above the bbox regardless of scroll
        }}
      >
        {annotation.data.text}
      </div>
    );
  }

  function renderEditor(props) {
    const { geometry } = props.annotation;
    if (!geometry) return null;

    return (
      <div
        style={{
          background: 'white',
          borderRadius: 3,
          position: 'absolute',
          left: `${geometry.x}%`,
          top: `${geometry.y + geometry.height}%`,
        }}
        className="p-2 mt-1 rounded shadow-lg"
      >
        <select
          onChange={(e) => handleAnnotationChange(e, props)}
          className={style.editorSelect}
        >
          <option value="">Select a tag</option>
          {tags.map((tag, index) => (
            <option key={index} value={tag}>{tag}</option>
          ))}
        </select>
      </div>
    );
  }

  const onChange = (newAnnotation) => {
    setAnnotation(newAnnotation);
  };

  const handleAnnotationChange = (e, props) => {
    const newAnnotation = {
      ...props.annotation,
      data: {
        ...props.annotation.data,
        text: e.target.value
      }
    };
    setAnnotation(newAnnotation);
    onSubmit(newAnnotation);
  };

  const onSubmit = (newAnnotation) => {
    const { geometry, data } = newAnnotation;
    redoStackRef.current = [];
    setAnnotation({});
    const newAnnotationData = {
      geometry,
      data: {
        ...data,
        id: Math.random(),
        imageId: selectedImage,
      },
    };
    setAnnotations([...annotations, newAnnotationData]);
    undoStackRef.current.push(annotations);
  };

  const handleImageSelection = (index) => {
    setSelectedImage(imagesPreview[index]);
  };

  const handleUndo = () => {
    if (undoStackRef.current.length > 0) {
      const lastAnnotations = undoStackRef.current.pop();
      redoStackRef.current.push([...annotations]);
      setAnnotations(lastAnnotations);
    }
  };

  const handleRedo = () => {
    if (redoStackRef.current.length > 0) {
      const nextAnnotations = redoStackRef.current.pop();
      undoStackRef.current.push([...annotations]);
      setAnnotations(nextAnnotations);
    }
  };

  const applyAnnotationsToNewImage = (previousAnnotations) => {
    return previousAnnotations.map(anno => ({
      geometry: anno.geometry,
      data: {
        ...anno.data,
        id: Math.random(),
        imageId: selectedImage,
      }
    }));
  };

  const convertImageToBase64 = (url) => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.onload = () => {
        const reader = new FileReader();
        reader.onloadend = () => {
          resolve(reader.result);
        };
        reader.readAsDataURL(xhr.response);
      };
      xhr.onerror = (e) => {
        reject(e);
      };
      xhr.open('GET', url);
      xhr.responseType = 'blob';
      xhr.send();
    });
  };

  const exportAnnotationsForYOLOv5 = async () => {
    const yolov5Annotations = await Promise.all(annotations.map(async (anno) => {
      const { geometry, data } = anno;
      const { x, y, width, height } = geometry;
      const { imageId, text } = data;
      const imgWidth = 1; // Update with actual image width
      const imgHeight = 1; // Update with actual image height
      const x_center = (x + (width / 2)) / imgWidth;
      const y_center = (y + (height / 2)) / imgHeight;
      const bbox_width = width / imgWidth;
      const bbox_height = height / imgHeight;

      const imageData = await convertImageToBase64(imageId);

      return {
        imageId,
        label: text,
        x_center,
        y_center,
        bbox_width,
        bbox_height,
        imageData // Ensure this contains base64 image data
      };
    }));

    try {
      const response = await axios.post('http://localhost:5000/save-annotations', {
        annotations: yolov5Annotations,
        datasetFolder,
        trainPercent,
        valPercent,
        testPercent,
        tags
      });
      alert(response.data.message);
    } catch (error) {
      console.error('Failed to save annotations:', error);
      alert('Failed to save annotations.');
    }
  };

  const startTraining = () => {
    setIsTraining(true);
    setTrainingCompleted(false);
    axios.get(`http://localhost:5000/start-training?dataset_folder=${datasetFolder}`)
      .then(response => {
        console.log('Training started');
        setTimeout(() => {
          setIsTraining(false);
          setTrainingCompleted(true);
        }, 5000); // Simulate training time
      })
      .catch(error => {
        console.error('Error starting training:', error);
        setIsTraining(false);
      });
  };

  const handleSaveAndContinue = () => {
    setShowSaveContinueModal(true);
  };

  const confirmSaveAndContinue = () => {
    setShowSaveContinueModal(false);
    setActiveTab('annotator');
  };

  return (
    <div className="px-4">
      <div className="flex flex-col items-center">
        <div className="w-full max-w-4xl mt-6">
          <div className="flex mb-4 space-x-4 border-b border-gray-300">
            <button
              className={`${style.tabButton} ${activeTab === 'loaderImage' ? style.tabButtonActive : style.tabButtonInactive}`}
              onClick={() => setActiveTab('loaderImage')}
            >
              Loader Image
            </button>
            <button
              className={`${style.tabButton} ${activeTab === 'annotator' ? style.tabButtonActive : style.tabButtonInactive}`}
              onClick={() => setActiveTab('annotator')}
            >
              Annotator
            </button>
            <button
              className={`${style.tabButton} ${activeTab === 'configTrain' ? style.tabButtonActive : style.tabButtonInactive}`}
              onClick={() => setActiveTab('configTrain')}
            >
              Config Train
            </button>
            <button
              className={`${style.tabButton} ${activeTab === 'resultTrain' ? style.tabButtonActive : style.tabButtonInactive}`}
              onClick={() => setActiveTab('resultTrain')}
            >
              Result Train
            </button>
            <button
              className={`${style.tabButton} ${activeTab === 'statistics' ? style.tabButtonActive : style.tabButtonInactive}`}
              onClick={() => setActiveTab('statistics')}
            >
              Statistics
            </button>
          </div>

          {activeTab === 'loaderImage' && (
            <div>
              <UploadImages
                imageFiles={imageFiles}
                setImageFiles={setImageFiles}
                imagesPreview={imagesPreview}
                setImagesPreview={setImagesPreview}
                tags={tags}
                setTags={setTags}
              />
              <button className={style.button} onClick={confirmSaveAndContinue} type="button">Save and Continue</button>
            </div>
          )}

          {activeTab === 'annotator' && (
            <div className="p-6 bg-white rounded-lg shadow-md">
              <p className="mb-6 text-2xl font-semibold text-center">Annotate Images</p>
              <div className={style.arrowContainer}>
                <button onClick={handlePrevImage} className={style.arrowButton}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <div className="flex items-center gap-4">
                  <button className={style.undoRedoButton} onClick={handleUndo} type="button">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8z"/>
                    </svg>
                  </button>
                  <button className={style.undoRedoButton} onClick={handleRedo} type="button">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 20v-3l4 4-4 4v-3c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6h2c0-4.41-3.59-8-8-8s-8 3.59-8 8 3.59 8 8 8z"/>
                    </svg>
                  </button>
                </div>
                <button onClick={handleNextImage} className={style.arrowButton}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
              <div className="w-full md:w-[600px] m-auto cursor-crosshair">
                {/* <Annotation
                  src={selectedImage}
                  alt="Annotate image"
                  annotations={annotations.filter((anno) => anno.data.imageId === selectedImage)}
                  value={annotation}
                  type={annotation.type}
                  className="h-[600px] border border-gray-300 rounded-lg shadow-sm"
                  onChange={onChange}
                  onSubmit={onSubmit}
                  allowTouch
                  renderOverlay={() => null}
                  renderSelector={renderSelector}
                  renderHighlight={renderHighlight}
                  renderContent={renderContent}
                  renderEditor={renderEditor}
                /> */}
              </div>
              <div className="flex items-center justify-between mt-4">
  <div className="flex items-center">
    <label className="block mr-2">
      Dataset Folder:
    </label>
    <input
      type="text"
      value={datasetFolder}
      onChange={(e) => setDatasetFolder(e.target.value)}
      className="p-1 border rounded"
    />
  </div>
  <button className={style.button} onClick={exportAnnotationsForYOLOv5} type="button">
    Save Annotations
  </button>
</div>
            </div>
          )}

          {activeTab === 'configTrain' && (
            <div>
              <p className="text-xl font-semibold text-center">Configure Training</p>
              <div className="flex items-center justify-center gap-4 my-4">
                <label className="block">
                  Train (%):
                  <input
                    type="number"
                    value={trainPercent}
                    onChange={(e) => setTrainPercent(parseInt(e.target.value))}
                    className="p-1 ml-2 border rounded"
                    min="0"
                    max="100"
                  />
                </label>
                <label className="block">
                  Validation (%):
                  <input
                    type="number"
                    value={valPercent}
                    onChange={(e) => setValPercent(parseInt(e.target.value))}
                    className="p-1 ml-2 border rounded"
                    min="0"
                    max="100"
                  />
                </label>
                <label className="block">
                  Test (%):
                  <input
                    type="number"
                    value={testPercent}
                    onChange={(e) => setTestPercent(parseInt(e.target.value))}
                    className="p-1 ml-2 border rounded"
                    min="0"
                    max="100"
                  />
                </label>
         
                <button className={style.button} onClick={() => setShowModal(true)} type="button">Start Training</button>
              </div>
            </div>
          )}

          {activeTab === 'resultTrain' && (
            <div>
              <p className="text-xl font-semibold text-center">Training Results</p>
              {isTraining ? (
                <div className="flex items-center justify-center gap-4 my-4">
                  <CircularProgress />
                  <p>Training in progress...</p>
                </div>
              ) : (
                trainingCompleted && <p className="text-xl font-semibold text-center">Training started successfully!</p>
              )}
            </div>
          )}

          {activeTab === 'statistics' && (
            <div>
              <p className="text-xl font-semibold text-center">Statistics</p>
              {/* Add your statistics content here */}
            </div>
          )}

          {showModal && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="p-6 bg-white rounded shadow-lg">
                <p>Are you sure you want to start training? Ensure all annotations are saved.</p>
                <div className="flex items-center justify-end mt-4">
                  <button className={style.modalButton} onClick={() => { setShowModal(false); startTraining(); }} type="button">Yes, start training</button>
                  <button className="ml-2 text-gray-500" onClick={() => setShowModal(false)} type="button">Cancel</button>
                </div>
              </div>
            </div>
          )}

          {showSaveContinueModal && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="p-6 bg-white rounded shadow-lg">
                <p>Are you sure you want to save and continue? Ensure all necessary steps are completed.</p>
                <div className="flex items-center justify-end mt-4">
                  <button className={style.modalButton} onClick={confirmSaveAndContinue} type="button">Yes, save and continue</button>
                  <button className="ml-2 text-gray-500" onClick={() => setShowSaveContinueModal(false)} type="button">Cancel</button>
                </div>
              </div>
            </div>
          )}

          {isTraining && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="p-6 bg-white rounded shadow-lg">
                <div className="flex items-center justify-center gap-4 my-4">
                  <CircularProgress />
                  <p>Training in progress...</p>
                </div>
              </div>
            </div>
          )}

          {trainingCompleted && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="p-6 bg-white rounded shadow-lg">
                <p>The training system has finished successfully!</p>
                <button className={style.modalButton} onClick={() => setTrainingCompleted(false)} type="button">Close</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Annotator;


