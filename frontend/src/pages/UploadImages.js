import React, { useState } from 'react';
import { FaTrash } from 'react-icons/fa';

const UploadImages = ({ imageFiles, setImageFiles, imagesPreview, setImagesPreview,tags, setTags }) => {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [tagInput, setTagInput] = useState('');
  const [displayMode, setDisplayMode] = useState('upload');
  const [showDelete, setShowDelete] = useState(new Array(imagesPreview.length).fill(false));
  const [hoverStates, setHoverStates] = useState({});

  const handleFileChange = (event) => {
    const selectedFiles = event.target.files;
    if (selectedFiles.length > 0) {
      processFiles(selectedFiles);
    }
  };

  const processFiles = (files) => {
    const validFiles = Array.from(files).filter(file =>
      ['image/png', 'image/jpeg', 'image/bmp'].includes(file.type)
    );

    if (validFiles.length > 0) {
      setIsLoading(true);
      setTimeout(() => {
        const newImageFiles = [...imageFiles, ...validFiles];
        const newImagePreviews = validFiles.map(file => URL.createObjectURL(file));
        setImageFiles(newImageFiles);
        setImagesPreview([...imagesPreview, ...newImagePreviews]);
        setIsLoading(false);
        setDisplayMode('gallery');
      }, 1000);
    } else {
      alert('No valid file types selected. Please select PNG, JPEG, or BMP files.');
    }
  };

  const handleTagInput = (event) => {
    const input = event.target.value;
    setTagInput(input);
    if (event.key === 'Enter' && input.trim() !== '') {
      setTags([...tags, input.trim()]);
      setTagInput('');
      event.preventDefault();
    }
  };

  const removeTag = (indexToRemove) => {
    setTags(tags.filter((_, index) => index !== indexToRemove));
  };

  const removeImage = (indexToRemove) => {
    setImagesPreview(imagesPreview.filter((_, index) => index !== indexToRemove));
    setImageFiles(imageFiles.filter((_, index) => index !== indexToRemove));
    setShowDelete(showDelete.filter((_, index) => index !== indexToRemove));
  };

  const handleSaveAndContinue = () => {
    console.log('Saving Project:', projectName);
    console.log('Saving Tags:', tags);
    console.log('Saving Images:', imageFiles);
    alert('Project, tags, and images have been saved!');
    // Here you would typically handle further processing, like sending data to a server
  };
  const toggleDeleteIcon = (index, isVisible) => {
    const updatedShowDelete = [...showDelete];
    updatedShowDelete[index] = isVisible;
    setShowDelete(updatedShowDelete);
  };
  const toggleHoverState = (index, isHovering) => {
    setHoverStates(prevStates => ({
        ...prevStates,
        [index]: isHovering
    }));
};

const renderImageGallery = () => (
  <div className="grid grid-cols-4 gap-4 p-4">
    {imagesPreview.map((image, index) => (
      <div
        key={index}
        className="relative overflow-hidden border rounded"
        onMouseEnter={() => toggleHoverState(index, true)}
        onMouseLeave={() => toggleHoverState(index, false)}
      >
        <img src={image} alt={`Uploaded ${index}`} className="w-full h-auto" />
        {hoverStates[index] && (
          <button
            onClick={() => removeImage(index)}
            className="absolute inset-0 p-2 m-auto bg-transparent rounded-full"
            style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }} // Center the button
          >
            <FaTrash style={{ color: 'red', fontSize: '24px' }} />
          </button>
        )}
      </div>
    ))}
    {/* <button
      onClick={handleSaveAndContinue}
      className="col-span-4 px-4 py-2 mt-4 text-white bg-blue-500 rounded hover:bg-blue-600"
    >
      Save and Continue
    </button> */}
  </div>
);


  const renderUploadInterface = () => (
    <div
      className={`p-10 border-2 border-dashed ${isDraggingOver ? 'border-purple-500' : 'border-gray-300'} relative text-center`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDraggingOver(true);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setIsDraggingOver(false);
        if (event.dataTransfer.files.length > 0) {
          processFiles(event.dataTransfer.files);
        }
      }}
      
    >
    {isLoading ? (
          <div className="flex items-center justify-center w-full h-full">
            <div className="w-12 h-12 mb-4 ease-linear border-4 border-t-4 border-gray-200 rounded-full loader"></div>
            <p className="text-xl font-semibold text-center">Loading...</p>
          </div>
        ) : (
          <p className="text-2xl font-bold">Drag and drop images and annotations to upload them.</p>
        )}  <div className="flex justify-center mt-4">
          <button
            className="px-4 py-2 font-bold text-white bg-purple-500 rounded hover:bg-purple-600"
            onClick={() => document.getElementById('file-input').click()}
          >
            Select Files
          </button>
          <button
            className="px-4 py-2 ml-4 font-bold text-white bg-gray-200 rounded hover:bg-gray-300"
            onClick={() => document.getElementById('folder-input').click()}
          >
            Select Folder
          </button>
        </div>
        <input
        id="file-input"
        type="file"
        multiple
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <input
        id="folder-input"
        type="file"
        webkitdirectory="true"
        directory=""
        multiple
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
    
  );

  return (
    <div className="max-w-5xl p-8 mx-auto bg-white rounded-lg shadow-md">
     

     <div className="mb-6">
  <div className="flex items-center mb-4 space-x-2">
    <label htmlFor="projectName" className="text-sm font-medium text-gray-700 min-w-[100px]">
      Project Name
    </label>
    <input
      id="projectName"
      type="text"
      placeholder="Enter the project name"
      value={projectName}
      onChange={e => setProjectName(e.target.value)}
      className="flex-1 p-2 border rounded"
      style={{ flex: '1 1 0%' }} // Ensures that the input takes the remaining space
    />
  </div>

  <div className="flex items-center space-x-2">
    <label htmlFor="tagsInput" className="text-sm font-medium text-gray-700 min-w-[100px]">
      Tags
    </label>
    <input
      id="tagsInput"
      type="text"
      placeholder="Add tags (press Enter to add)"
      value={tagInput}
      onChange={e => setTagInput(e.target.value)}
      onKeyPress={handleTagInput}
      className="flex-1 p-2 border rounded"
      style={{ flex: '1 1 0%' }} // Matches the project name input size
    />
  </div>


        <div className="flex flex-wrap gap-2 mt-2">
          {tags.map((tag, index) => (
            <div key={index} className="flex items-center px-2 bg-purple-200 rounded">
              {tag}
              <button onClick={() => removeTag(index)} className="ml-2 text-sm text-purple-500 hover:text-purple-700">&#10005;</button>
            </div>
          ))}
        </div>
      </div>
      {displayMode === 'upload' ? renderUploadInterface() : renderImageGallery()}
    </div>
  );
};

export default UploadImages;
