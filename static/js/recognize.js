const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
let video;

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;
    video.srcObject = stream;
    video.style.display = "none";
    document.body.appendChild(video);

    requestAnimationFrame(draw);
    setInterval(recognizeFaces, 1000);
  });

let boxes = [];
let names = [];

function draw() {
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  for (let i = 0; i < boxes.length; i++) {
    const [top, right, bottom, left] = boxes[i];
    context.strokeStyle = names[i] === "Inconnu" ? "red" : "green";
    context.lineWidth = 2;
    context.strokeRect(left, top, right - left, bottom - top);
    context.fillStyle = context.strokeStyle;
    context.fillText(names[i], left, top - 5);
  }
  requestAnimationFrame(draw);
}

function recognizeFaces() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
  tempCanvas.toBlob(blob => {
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    fetch("/recognize", {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      boxes = data.boxes;
      names = data.names;
    });
  }, "image/jpeg");
}
