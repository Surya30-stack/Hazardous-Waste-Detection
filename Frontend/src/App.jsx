import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const [cameraActive, setCameraActive] = useState(false)
  const [cameraLoading, setCameraLoading] = useState(false)
  const [isLiveDetection, setIsLiveDetection] = useState(false)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const liveDetectionRef = useRef(null)

  // =========================
  // IMAGE UPLOAD
  // =========================

  const handleFileChange = (event) => {
    const file = event.target.files[0]

    if (!file) {
      return
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setAnnotatedImage(null)
    setError(null)
  }

  // =========================
  // IMAGE DETECTION
  // =========================

  const handleDetect = async () => {
    if (!selectedFile) {
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setAnnotatedImage(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(
        'http://127.0.0.1:8000/predict',
        {
          method: 'POST',
          body: formData
        }
      )

      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status}`
        )
      }

      const data = await response.json()

      setResult(data)
      setAnnotatedImage(data.annotated_image_base64)

    } catch (error) {
      console.error('Image detection error:', error)

      setError(
        'Unable to process the image. Please make sure the FastAPI server is running.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  // =========================
  // CAMERA START
  // =========================

  const startCamera = async () => {
    try {
      setError(null)
      setCameraLoading(true)

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        })

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      setCameraActive(true)

    } catch (error) {
      console.error('Camera error:', error)

      setError(
        'Unable to access the camera. Please allow camera permission in your browser.'
      )
    } finally {
      setCameraLoading(false)
    }
  }

  // =========================
  // CAMERA STOP
  // =========================

  const stopCamera = () => {
    // Stop live detection first
    setIsLiveDetection(false)

    if (liveDetectionRef.current) {
      clearInterval(liveDetectionRef.current)
      liveDetectionRef.current = null
    }

    // Stop camera stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(
        (track) => track.stop()
      )

      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setCameraActive(false)
  }

  // =========================
  // SINGLE FRAME DETECTION
  // =========================

  const captureAndDetect = async () => {
    if (!videoRef.current || !cameraActive) {
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setAnnotatedImage(null)

    const video = videoRef.current
    const canvas = canvasRef.current

    if (!canvas) {
      setError('Unable to capture camera frame.')
      setIsLoading(false)
      return
    }

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      setError('Camera frame is not ready yet.')
      setIsLoading(false)
      return
    }

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const context = canvas.getContext('2d')

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    )

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setError(
            'Unable to create image from camera frame.'
          )
          setIsLoading(false)
          return
        }

        const formData = new FormData()

        formData.append(
          'file',
          blob,
          'webcam_capture.jpg'
        )

        try {
          const response = await fetch(
            'http://127.0.0.1:8000/predict',
            {
              method: 'POST',
              body: formData
            }
          )

          if (!response.ok) {
            throw new Error(
              `Server error: ${response.status}`
            )
          }

          const data = await response.json()

          setResult(data)
          setAnnotatedImage(
            data.annotated_image_base64
          )

        } catch (error) {
          console.error(
            'Webcam detection error:',
            error
          )

          setError(
            'Unable to process the webcam image. Please make sure the FastAPI server is running.'
          )
        } finally {
          setIsLoading(false)
        }
      },
      'image/jpeg',
      0.9
    )
  }

  // =========================
  // LIVE DETECTION START
  // =========================

  const startLiveDetection = () => {
    if (!cameraActive) {
      return
    }

    setError(null)
    setIsLiveDetection(true)
  }

  // =========================
  // LIVE DETECTION STOP
  // =========================

  const stopLiveDetection = () => {
    setIsLiveDetection(false)

    if (liveDetectionRef.current) {
      clearInterval(liveDetectionRef.current)
      liveDetectionRef.current = null
    }
  }

  // =========================
  // LIVE DETECTION ENGINE
  // =========================

  useEffect(() => {
    if (!isLiveDetection || !cameraActive) {
      return
    }

    const detectFrame = async () => {
      if (
        !videoRef.current ||
        !canvasRef.current
      ) {
        return
      }

      const video = videoRef.current
      const canvas = canvasRef.current

      if (
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {
        return
      }

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const context = canvas.getContext('2d')

      context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      )

      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            return
          }

          const formData = new FormData()

          formData.append(
            'file',
            blob,
            'live_webcam.jpg'
          )

          try {
            const response = await fetch(
              'http://127.0.0.1:8000/predict',
              {
                method: 'POST',
                body: formData
              }
            )

            if (!response.ok) {
              return
            }

            const data = await response.json()

            setResult(data)
            setAnnotatedImage(
              data.annotated_image_base64
            )

          } catch (error) {
            console.error(
              'Live detection error:',
              error
            )
          }
        },
        'image/jpeg',
        0.8
      )
    }

    // Run detection approximately once every second
    liveDetectionRef.current = setInterval(
      detectFrame,
      1000
    )

    return () => {
      if (liveDetectionRef.current) {
        clearInterval(
          liveDetectionRef.current
        )

        liveDetectionRef.current = null
      }
    }

  }, [isLiveDetection, cameraActive])

  // =========================
  // CLEAR UPLOAD / RESULTS
  // =========================

  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setAnnotatedImage(null)
    setError(null)
  }

  // =========================
  // CLEANUP
  // =========================

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(
          (track) => track.stop()
        )
      }

      if (liveDetectionRef.current) {
        clearInterval(
          liveDetectionRef.current
        )
      }
    }
  }, [])

  // =========================
  // UI
  // =========================

  return (
    <div className="app">

      {/* Header */}

      <header className="header">

        <div className="header-content">

          <div className="logo">
            HW
          </div>

          <div>

            <h1>
              Hazardous Waste Detection
            </h1>

            <p>
              AI-powered waste detection using YOLO
            </p>

          </div>

        </div>

      </header>


      {/* Main Content */}

      <main className="container">

        {/* =========================
            UPLOAD CARD
        ========================= */}

        <section className="card upload-card">

          <div className="section-heading">

            <h2>
              Upload Image
            </h2>

            <p>
              Upload an image to detect hazardous waste
              using the trained YOLO model.
            </p>

          </div>


          <label className="file-input">

            <span>
              {selectedFile
                ? 'Choose another image'
                : 'Choose an image'}
            </span>

            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
            />

          </label>


          {selectedFile && (

            <div className="file-info">

              <strong>
                Selected file:
              </strong>{' '}

              {selectedFile.name}

            </div>

          )}


          {previewUrl && (

            <div className="image-card">

              <h3>
                Original Image
              </h3>

              <img
                src={previewUrl}
                alt="Selected image"
              />

            </div>

          )}


          {selectedFile && (

            <div className="button-group">

              <button
                className="detect-button"
                onClick={handleDetect}
                disabled={isLoading}
              >
                {isLoading
                  ? 'Detecting...'
                  : 'Detect Image'}
              </button>


              <button
                className="clear-button"
                onClick={handleClear}
                disabled={isLoading}
              >
                Clear
              </button>

            </div>

          )}


          {error && (

            <div className="error-message">
              {error}
            </div>

          )}

        </section>


        {/* =========================
            CAMERA CARD
        ========================= */}

        <section className="card camera-card">

          <div className="section-heading">

            <h2>
              Webcam Detection
            </h2>

            <p>
              Capture an image using your camera or
              run live hazardous waste detection.
            </p>

          </div>


          <div className="camera-container">

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
            />

          </div>


          <canvas
            ref={canvasRef}
            style={{ display: 'none' }}
          />


          <div className="button-group">

            {!cameraActive ? (

              <button
                className="detect-button"
                onClick={startCamera}
                disabled={cameraLoading}
              >
                {cameraLoading
                  ? 'Starting Camera...'
                  : 'Start Camera'}
              </button>

            ) : (

              <>

                {!isLiveDetection && (

                  <button
                    className="detect-button"
                    onClick={startLiveDetection}
                  >
                    Start Live Detection
                  </button>

                )}


                {isLiveDetection && (

                  <button
                    className="clear-button"
                    onClick={stopLiveDetection}
                  >
                    Stop Live Detection
                  </button>

                )}


                {!isLiveDetection && (

                  <button
                    className="detect-button"
                    onClick={captureAndDetect}
                    disabled={isLoading}
                  >
                    {isLoading
                      ? 'Detecting...'
                      : 'Capture & Detect'}
                  </button>

                )}


                <button
                  className="clear-button"
                  onClick={stopCamera}
                >
                  Stop Camera
                </button>

              </>

            )}

          </div>

        </section>


        {/* =========================
            RESULTS
        ========================= */}

        {result && (

          <section className="card result-card">

            <div className="section-heading">

              <h2>
                Detection Results
              </h2>

              <p>
                Results generated by the YOLO detection model.
              </p>

            </div>


            {/* Annotated Image */}

            {annotatedImage && (

              <div className="image-card">

                <h3>
                  Detected Image
                </h3>

                <img
                  src={"data:image/jpeg;base64," + annotatedImage}
                  alt="YOLO detection result"
                />

              </div>

            )}


            {/* Summary */}

            <div className="summary">

              <div className="summary-box">

                <span className="summary-label">
                  Objects Detected
                </span>

                <span className="summary-value">
                  {result.detection_count}
                </span>

              </div>

            </div>


            {/* Detection Details */}

            <div className="detections">

              <h3>
                Detection Details
              </h3>


              {result.detections &&
              result.detections.length > 0 ? (

                result.detections.map(
                  (detection, index) => (

                    <div
                      className="detection-item"
                      key={index}
                    >

                      <div>

                        <span className="detail-label">
                          Class
                        </span>

                        <strong>
                          {detection.class_name}
                        </strong>

                      </div>


                      <div>

                        <span className="detail-label">
                          Confidence
                        </span>

                        <strong>
                          {(detection.confidence * 100).toFixed(2)}%
                        </strong>

                      </div>

                    </div>

                  )
                )

              ) : (

                <p className="no-detection">
                  No hazardous waste detected.
                </p>

              )}

            </div>

          </section>

        )}

      </main>


      {/* Footer */}

      <footer className="footer">

        <p>
          Hazardous Waste Detection • YOLO + FastAPI + React
        </p>

      </footer>

    </div>
  )
}

export default App
