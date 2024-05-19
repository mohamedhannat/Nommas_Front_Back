import React, { useEffect,useState, useRef } from 'react';
import Annotation from 'react-image-annotation';
import image1 from "./image1.png";
import UploadImages from './UploadImages';

const Annotator = ({ imageFiles, setImageFiles, imagesPreview, setImagesPreview , tags,setTags}) => {

  const [annotations, setAnnotations] = useState([]);
  const [annotation, setAnnotation] = useState({});
  const [selectedImage, setSelectedImage] = useState(image1);
  const undoStackRef = useRef([]);
  const redoStackRef = useRef([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const style = {
    button: "text-[#fff] bg-[#4ca3dd] py-[2px] px-2 rounded-[5px]",
  }
  // Function to generate color based on tag
  // Define a wider range of colors
  const colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', '#FF4500', '#DA70D6', '#40E0D0', '#FF69B4', '#BA55D3'];


  useEffect(() => {
    // Update selected image when currentIndex changes
    setSelectedImage(imagesPreview[currentIndex]);
    setAnnotations([]); // Clear annotations when changing images
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
    if (!geometry) return null

    return (
      <Box
        geometry={geometry}
        style={{
          border: `solid 2px #E10000`
        }}
      />
    )
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
        className="p-2 rounded-[10px] mt-[5px]"
      >
        <select
          onChange={(e) => handleAnnotationChange(e, props)}
          className="block w-full p-2 mt-1 focus:outline-none"
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
  }

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
    return previousAnnotations
      .map(anno => ({
        geometry: anno.geometry,
        data: {
          ...anno.data,
          id: Math.random(),
          imageId: selectedImage,
        }
      }));
  };
  
  const applyAnnotationsToSelectedImage = () => {
    const annotationsForNewImage = applyAnnotationsToNewImage(annotations);
    setAnnotations([...annotations, ...annotationsForNewImage]);
  };

  const exportAnnotationsForYOLOv7 = () => {
    const yolov7Annotations = annotations.map(anno => {
      const { geometry, data } = anno;
      const { x, y, width, height } = geometry;
      // Assuming the label is stored in data.text
      return {
        image_path: data.imageId, // This should be the path or identifier for the image
        annotations: [
          {
            class_label: data.text,
            x_center: x + (width / 2),
            y_center: y + (height / 2),
            bbox_width: width,
            bbox_height: height
          }
        ]
      };
    });
  
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(yolov7Annotations));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "yolov7_annotations.json");
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };
  
  return (
    <div className="px-4">
      <div className="flex flex-wrap items-start gap-4 justify-evenly">

        <div className="mt-6">
          <p className="text-center text-[20px] font-[600]">Annotate images</p>
          <div className="flex items-center justify-center gap-4 my-4">
            <button className={style.button} onClick={handleUndo} type="button">Undo</button>
            <button className={style.button} onClick={handleRedo} type="button">Redo</button>
            <button className={style.button} onClick={applyAnnotationsToSelectedImage} type="button">Apply all</button>
            <button className={style.button} onClick={exportAnnotationsForYOLOv7} type="button">Export for YOLOv7</button>

          </div>
          <div className="w-full md:w-[600px] m-auto cursor-crosshair">
            <Annotation
              src={selectedImage}
              alt="Annotate image"
              annotations={annotations.filter((anno) => anno.data.imageId === selectedImage)}
              value={annotation}
              type={annotation.type}
              className="h-[600px]"
              onChange={onChange}
              onSubmit={onSubmit}
              allowTouch
              renderOverlay={() => null}
              renderSelector={renderSelector}
              renderHighlight={renderHighlight}
              renderContent={renderContent}
              renderEditor={renderEditor}
            />
          </div>
          <div className="mt-[4%] flex flex-wrap gap-4 items-center justify-center mb-4">
            {imageFiles?.map((file, index) => (
              <div key={index} className="h-[70px]">
                <img
                  src={imagesPreview[index]}
                  onClick={() => handleImageSelection(index)}
                  alt={file.name}
                  className="w-[100px] h-full cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Annotator;
