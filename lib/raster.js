paper.install(window);

const typeInput = document.querySelector("#typeInput");
typeInput.focus();
const typeInputContents = typeInput.value;
const enterInput = (e) => {
  typeInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      const wrapper = document.querySelector(".wrap-log");
      const text = document.createElement("p");
      text.innerHTML = e.target.value;

      // text.style.left =`${Math.random() * 100}vw`
      // text.style.top = `${Math.random() * 100}vh`

      wrapper.appendChild(text);

      e.target.value = "";

      console.log(e.target.value);
    }
  });
};
const init = () => {
  enterInput();
};

init();

// Upload image
const defaultBtn = document.querySelector("#default-btn");
const customBtn = document.querySelector("#custom-btn");
const img = document.querySelector("img");

function defaultBtnActive() {
  defaultBtn.click();
}

// Image raster
// Get a reference to the canvas object
var canvas = document.getElementById("canvas");
// Create an empty project and a view for the canvas:
paper.setup(canvas);

defaultBtn.addEventListener("change", function () {
  const file = this.files[0];

  if (file) {
    const reader = new FileReader();
    reader.onload = function () {
      const result = reader.result;
      img.src = result;
      localStorage.setItem("image", result);

      var raster = new Raster("image");
      var loaded = false;

      raster.on("load", function () {
        console.log("Load raster");
        loaded = true;
        onResize();
      });

      raster.visible = false;

      var lastValue = typeInput.value;
      function handleResize(event) {
        if (!loaded) return;
        console.log(typeInput.value, lastValue);
        if (typeInput.value === lastValue) return;
        lastValue = typeInput.value;
        var size = this.bounds.size.clone();

        size.width /= 2;

        var path = new Path.Rectangle({
          point: this.bounds.topLeft.floor(),
          size: size.ceil(),
          onFrame: handleResize,
        });
        path.fillColor = raster.getAverageColor(path);

        var path = new Path.Rectangle({
          point: this.bounds.topCenter.ceil(),
          size: size.floor(),
          onFrame: handleResize,
        });
        path.fillColor = raster.getAverageColor(path);
        this.remove();
      }

      function onResize(event) {
        if (!loaded) return;
        project.activeLayer.removeChildren();

        raster.fitBounds(view.bounds, true);

        var path = new Path.Rectangle({
          rectangle: view.bounds,
          fillColor: raster.getAverageColor(view.bounds),
          onFrame: handleResize,
        });
      }
    };
    reader.readAsDataURL(file);
    view.draw();
  }
});
