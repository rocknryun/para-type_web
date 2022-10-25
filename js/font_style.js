const about = document.querySelector(".about");
const selectfont = document.querySelector(".SelectFont");
let font

function setfont(event) {
    console.log(about.style.fontFamily, selectfont.value);
    about.style.fontFamily=selectfont.value;
}
console.log(selectfont, about)

slider.oninput = () => {
    const aboutCol = document.getElementById('#about-col')
    const slider = document.getElementById('#slider');
    const pixels = document.getElementById('.pixels');
    pixels.innerHTML = slider.value;
    
    document.querySelector.aboutCol.style.setProperty('--font-size','${slider.value)px');

}
console.log(slider, pixels)
