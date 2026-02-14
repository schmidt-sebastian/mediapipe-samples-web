
const fs = require('fs');
const path = require('path');

const imagePath = path.join(__dirname, 'public/dog.jpg');
const templatePath = path.join(__dirname, 'src/templates/image-segmentation.html');

try {
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString('base64');
    const dataUri = `data:image/jpeg;base64,${base64Image}`;

    let templateContent = fs.readFileSync(templatePath, 'utf8');
    
    // Replace src="dog.jpg" with the data URI
    // We look for src="dog.jpg" specifically
    const newContent = templateContent.replace('src="dog.jpg"', `src="${dataUri}"`);

    fs.writeFileSync(templatePath, newContent);
    console.log('Successfully inlined dog.jpg into image-segmentation.html');
} catch (error) {
    console.error('Error inlining image:', error);
    process.exit(1);
}
