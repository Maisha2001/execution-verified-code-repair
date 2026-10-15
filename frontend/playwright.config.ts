import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',workers:1,timeout:30000,use:{baseURL:'http://127.0.0.1:8019',browserName:'chromium'},
 webServer:{command:`"${process.env.REPAIR_PYTHON || 'python'}" ../scripts/serve_ui_fixture.py`,url:'http://127.0.0.1:8019/api/health',timeout:30000},
 reporter:[['list'],['json',{outputFile:'test-results/results.json'}]]});
