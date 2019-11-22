function createTestTable(nrows,ncols) {
  let outputTable = document.getElementById("test-table");
  for (let i = 0; i < nrows; i++) {
    let tr = outputTable.insertRow();
    for(let j = 0; j < ncols; j++) {
      let td = tr.insertCell();
      td.style.border = '1px solid black';
      //canvas for the digit image
      const canvas1 = document.createElement('canvas');
      canvas1.width = 28*2;
      canvas1.height = 28*2+12;
      canvas1.style = "z-index:1; left:0px; top:0px;"
      td.appendChild(canvas1);
    }
  }
}


function draw(data, canvas) {
  const [width, height] = [28, 28];
  // canvas.width = width*2;
  // canvas.height = height*2;
  const ctx = canvas.getContext('2d');

  const imageData = new ImageData(width, height);
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    let val = (1-data[i]) * 255;
    imageData.data[j + 0] = val;
    imageData.data[j + 1] = val;
    imageData.data[j + 2] = val;
    imageData.data[j + 3] = 255-val;

  }
  // ctx.putImageData(imageData, 0, 0);
  let tempCanvas = new OffscreenCanvas(width,height);
  tempCanvas.getContext('2d').putImageData(imageData,0,0);
  ctx.drawImage(tempCanvas,0,0,width*2,height*2);
}


function drawTable(table,d) {
  let i = 0;
  for (let cell of Array.from(table.rows).flatMap(row => Array.from(row.cells))) {
    draw(d.slice([i],[1]).dataSync(),cell.children[0]);
    ++i;
  }
}

function clearTable(table) {
  let i = 0;
  for (let cell of Array.from(table.rows).flatMap(row => Array.from(row.cells))) {
    let canvas = cell.children[0];
    ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ++i;
  }
}

function setStatus(text) {
  document.getElementById("status").innerHTML = "<b>CNN Model status</b><br>"+text;
}

function setTrainingStatus(loss,i) {
  setStatus(`Training batch: ${i}, Current test loss: ${loss[0].toFixed(2)}`)
}

var data;
async function load() {
  let dataModule = await import('./data.js');
  MnistData = dataModule.MnistData;
  data = new MnistData();
  await data.load();
}

var modelModule;
var micp;
async function mnist() {
  setStatus("Loading...")
  await load();
  modelModule = await import('./model.js');
  MICP = modelModule.MICP;
  train = modelModule.train;
  await train(data,setTrainingStatus);
  calData = data.nextCalibrBatch(2000);       // Limited by WebGL
  calScores = modelModule.predictSoft(calData.xs);
  micp = new MICP(calScores, calData.labels);
  testData = data.nextTestBatch(100)
  testPreds = modelModule.predictSoft(testData.xs)
  pVals = micp.pValues(testPreds);
  drawTable(document.getElementById("test-table"),
            testData.xs);
}

function newTestData() {
  testData = data.nextTestBatch(100);
  testPreds = modelModule.predictSoft(testData.xs);
  pVals = micp.pValues(testPreds);
  let table = document.getElementById("test-table");
  clearTable(table);
  drawTable(table, testData.xs);
  cpRecalc();
}

async function trainMore() {
  await train(data,setTrainingStatus);
  calData = data.nextCalibrBatch(2000);       // Limited by WebGL
  calScores = modelModule.predictSoft(calData.xs);
  micp = new MICP(calScores, calData.labels);
  testPreds = modelModule.predictSoft(testData.xs)
  pVals = micp.pValues(testPreds);
  cpRecalc();
}

async function cpRecalc() {  // Do I need to tidy() this?
  eps = parseFloat(document.getElementById('epsilonInput').value);
  preds = pVals.greater(eps);
  labels = testData.labels;
  correct = tf.any(tf.logicalAnd(preds,labels), 1)
              .dataSync();
  predsArray = preds.dataSync();
  // update table
  i = 0;
  let table = document.getElementById("test-table");
  for (let cell of Array.from(table.rows).flatMap(row => Array.from(row.cells))) {
    if (correct[i]) {
      cell.style.background = 'green';
    } else {
      cell.style.background = 'red';
    }

    let ctx = cell.children[0].getContext('2d');
    ctx.clearRect(0,28*2,28*2,10);            // remove hardcode numbers
    ctx.font = "10px system-ui";
    for (let j=0; j<10; j++) {        // remove hardcode numbers
      if (predsArray[i*10+j]) {       // remove hardcode numbers
        ctx.fillStyle = 'yellow';
        //ctx.fillRect(3+5*j,2,4,4);
        ctx.fillText(`${j}`,3+5*j,28*2+8); // remove hardcode numbers
      }
    }
    ++i;
  }
  totalCorrect = correct.reduce((x,y) => x+y);
  let errorRate = (correct.length - totalCorrect) / correct.length;

  document.getElementById("errorRate").innerHTML = errorRate.toFixed(2);

  let predsSetSizes = preds.sum(1);
  let mask = predsSetSizes.greater(0);
  let predSetSizesFiltered = await tf.booleanMaskAsync(predsSetSizes,mask);
  count = mask.sum().dataSync()[0];
  if (count==0) {
    avgPredSetSize = 0;
  } else {
    avgPredSetSize = predSetSizesFiltered.sum().dataSync()[0] / count;
  }

  document.getElementById("avgPredSetSize").innerHTML = avgPredSetSize.toFixed(2);
}

createTestTable(10,10);

mnist().then(() => {
  cpRecalc();
  document.getElementById('epsilonInput').addEventListener('input',cpRecalc);
});
