import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function downloadFile(fileName: string, url: string): Promise<string> {
    const downloadPath = path.resolve(__dirname, 'assets');
    if (!fs.existsSync(downloadPath)) {
        fs.mkdirSync(downloadPath, { recursive: true });
    }
    const filePath = path.join(downloadPath, fileName);
    
    // Check if file already exists to avoid re-downloading
    if (fs.existsSync(filePath)) {
        return filePath;
    }

    console.log(`Downloading ${fileName} from ${url}...`);

    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filePath);
        https.get(url, (response) => {
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download ${url}: ${response.statusCode}`));
                return;
            }
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                resolve(filePath);
            });
        }).on('error', (err) => {
            fs.unlink(filePath, () => {}); // Delete the file async.
            reject(err);
        });
    });
}

export default async function globalSetup() {
    const assets = [

        {
            name: 'efficientdet_lite0.tflite',
            url: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'
        },
        {
            name: 'efficientdet_lite2.tflite',
            url: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite'
        },
        {
            name: 'deeplab_v3.tflite',
            url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite'
        },
        {
            name: 'hair_segmenter.tflite',
            url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite'
        },
        {
            name: 'yamnet.tflite',
            url: 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite'
        }
    ];

    console.log('--- Playwright Global Setup: Fetching Required Test Assets ---');
    try {
        await Promise.all(assets.map(asset => downloadFile(asset.name, asset.url)));
        console.log('--- Test Assets Verification Complete ---');
    } catch (error) {
        console.error('Error during global setup test asset fetch:', error);
        throw error;
    }
}
