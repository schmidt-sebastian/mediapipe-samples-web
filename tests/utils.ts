import { Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

export async function downloadTestImage(fileName: string, url: string): Promise<string> {
    const downloadPath = path.join(process.cwd(), 'tests', 'assets');
    if (!fs.existsSync(downloadPath)) {
        fs.mkdirSync(downloadPath, { recursive: true });
    }
    const filePath = path.join(downloadPath, fileName);
    
    // Check if file already exists to avoid re-downloading
    if (fs.existsSync(filePath)) {
        return filePath;
    }

    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filePath);
        https.get(url, (response) => {
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                resolve(filePath);
            });
        }).on('error', (err) => {
            fs.unlink(filePath, () => {}); // Delete the file async. (But we don't check result)
            reject(err);
        });
    });
}

export async function checkConsoleErrors(page: Page) {
    const errors: string[] = [];
    page.on('console', msg => {
        if (msg.type() === 'error') {
            errors.push(msg.text());
        }
    });
    return errors;
}
