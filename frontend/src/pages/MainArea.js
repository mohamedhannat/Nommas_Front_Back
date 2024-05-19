import React, { useState } from 'react';
import Annotator from "./Annotator";
import UploadImages from './UploadImages';
import Matcher from './Matching_Sift_Knn';
import VideoStream from './Stream';
import Detection from './Yolo_test';

function MainArea({ selectedMenu }) {
    const [imageFiles, setImageFiles] = useState([]);
    const [imagesPreview, setImagesPreview] = useState([]);
    const [tags, setTags] = useState([]);

    const renderContent = () => {
        switch (selectedMenu) {
     
            case 'train':
                return <Annotator 
                         imageFiles={imageFiles} 
                         setImageFiles={setImageFiles} 
                         imagesPreview={imagesPreview} 
                         setImagesPreview={setImagesPreview}
                         tags={tags}
                         setTags={setTags} />;
            case 'match':
                return <Matcher/>;
            case 'home':
                return <div>Welcome to Nossam App</div>;
            case 'settings':
                return <div>Settings Component</div>;
            case 'test':
                return <Detection/>;
            default:
                return <div>Welcome to Roboflow Clone</div>;
        }
    };

    return (
        <div className="flex items-center justify-center flex-grow p-8">
            <div className="w-full max-w-5xl mx-auto mt-8">
                {renderContent()}
            </div>
        </div>
    );
}

export default MainArea;
