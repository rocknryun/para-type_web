let rectWidth;

function setup() {
  createCanvas(innerWidth, innerHeight);
  noStroke();
  background(230);
  rectWidth = width / 4;
}

function draw() {
  // draw()를 여기에 작성하여 키보드 입력을 기다리는 동안 반복되게 합니다.
}

function keyPressed() {
  let keyIndex = -1;
  if (key >= 'a' && key <= 'z') {
    keyIndex = key.charCodeAt(0) - 'a'.charCodeAt(0);
  }
  if (keyIndex === -1) {
    // 글자 자판이 아닐 경우, 화면을 비웁니다.
    background(230);
  } else {
    // 글자 자판일 경우, 사각면을 채웁니다.
    randFill_r = Math.floor(Math.random() * 255 + 1);
    randFill_g = Math.floor(Math.random() * 255 + 1);
    randFill_b = Math.floor(Math.random() * 255 + 1);
    fill(randFill_r, randFill_g, randFill_b);
    let x = map(keyIndex, 0, 25, 0, width - rectWidth);
    rect(x, 0, rectWidth, height);
  }
}
