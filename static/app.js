// Import required MediaPipe libraries
import {
    HandLandmarker,
    FilesetResolver
  } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
  
  import {
    drawConnectors,
    drawLandmarks
  } from "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.8.0";
  
  import { HAND_CONNECTIONS } from "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.8.0";
  
  
  // Get DOM elements
  const demosSection = document.getElementById("demos");
  const video = document.getElementById("webcam");
  const canvasElement = document.getElementById("output_canvas");
  const canvasCtx = canvasElement.getContext("2d");
  const enableWebcamButton = document.getElementById("webcamButton");
  
  // Variables for hand landmark detection
  let handLandmarker = undefined;
  let runningMode = "IMAGE";
  let webcamRunning = false;
  
  // Asynchronous function to load and initialize the HandLandmarker
  const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
  
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU",
      },
      runningMode: runningMode,
      numHands: 2,
    });
  
    // Show the demos section when the model is ready
    demosSection.classList.remove("invisible");
  };
  createHandLandmarker();
  
  /********************************************************************
  // Demo 1: Detect hands in static images on click
  ********************************************************************/
  
  // Add event listeners to all clickable images
  const imageContainers = document.getElementsByClassName("detectOnClick");
  
  for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleClick);
  }
  
  // Handle image click and detect hand landmarks
  async function handleClick(event) {
    if (!handLandmarker) {
      console.log("Wait for handLandmarker to load before clicking!");
      return;
    }
  
    if (runningMode === "VIDEO") {
      runningMode = "IMAGE";
      await handLandmarker.setOptions({ runningMode: "IMAGE" });
    }
  
    // Remove existing landmarks
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (let i = allCanvas.length - 1; i >= 0; i--) {
      const n = allCanvas[i];
      n.parentNode.removeChild(n);
    }
  
    // Detect hand landmarks in the clicked image
    const handLandmarkerResult = await handLandmarker.detect(event.target);
  
    // Create a canvas overlay for drawing landmarks
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.width = event.target.naturalWidth;
    canvas.height = event.target.naturalHeight;
  
    event.target.parentNode.appendChild(canvas);
    const cxt = canvas.getContext("2d");
  
    // Draw landmarks and connections
    for (const landmarks of handLandmarkerResult.landmarks) {
      drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5,
      });
      drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
    }
  }
  
  /********************************************************************
  // Demo 2: Real-time hand detection with webcam
  ********************************************************************/
  
  // Check if webcam access is supported
  const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;
  
  if (hasGetUserMedia()) {
    enableWebcamButton.addEventListener("click", enableCam);
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
  
  // Enable webcam and start hand detection
  function enableCam() {
    if (!handLandmarker) {
      console.log("Wait! HandLandmarker not loaded yet.");
      return;
    }
  
    if (webcamRunning) {
      webcamRunning = false;
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
      webcamRunning = true;
      enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  
      // Start webcam stream
      const constraints = { video: true };
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
      });
    }
  }
  
  let lastVideoTime = -1;
  let results = undefined;
  
  // Predict landmarks in webcam feed
  async function predictWebcam() {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
  
    const startTimeMs = performance.now();
  
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      results = handLandmarker.detectForVideo(video, startTimeMs);
    }
  
    // Clear previous drawings
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
    // Draw detected landmarks
    if (results.landmarks) {
      for (const landmarks of results.landmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5,
        });
        drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
      }
    }
  
    canvasCtx.restore();
  
    // Continue detection if webcam is running
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
  }
  