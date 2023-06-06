"use strict";

const fileInput = document.getElementById('fileInput');
fileInput.onchange = () => {
    let selectedFile = fileInput.files[0];
    document.getElementById("fileForm").submit()
    // console.log(selectedFile);
}

