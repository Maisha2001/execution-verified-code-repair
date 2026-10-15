import {test,expect} from '@playwright/test';
test('failure evidence, safe text, export and persisted history',async({page})=>{
 await page.goto('/');await expect(page.getByRole('button',{name:'Run repair'})).toBeEnabled();
 await page.getByRole('button',{name:'Run repair'}).click();
 await expect(page.locator('#status')).toHaveText('completed');
 await expect(page.locator('#summary')).toContainText('No repair was accepted');
 await page.getByText('Raw model response',{exact:true}).click();
 await expect(page.locator('#attempts pre').last()).toContainText('<img');
 expect(await page.evaluate(()=>Object.hasOwn(window,'injected'))).toBe(false);
 await expect(page.locator('#export')).toBeVisible();
 const href=await page.locator('#export').getAttribute('href');expect((await page.request.get(href!)).status()).toBe(200);
 await page.reload();await expect(page.locator('#status')).toHaveText('completed');
});
test('cancel during a running job prevents acceptance',async({page})=>{
 await page.goto('/');await expect(page.getByRole('button',{name:'Run repair'})).toBeEnabled();
 await page.getByRole('button',{name:'Run repair'}).click();await expect(page.locator('#status')).toHaveText('running');
 await page.getByRole('button',{name:'Cancel repair'}).click();await expect(page.locator('#status')).toHaveText('cancelled');
 await expect(page.locator('#attempts')).toContainText('not ranked');
 await expect(page.locator('#summary')).toContainText('No repair was accepted');
 await expect(page.getByRole('button',{name:'Run repair'})).toBeEnabled();
});
test('mobile layout fits and controls retain accessible labels',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('/');
 await expect(page.getByLabel('Development example')).toHaveValue('gcd');
 await expect(page.getByLabel('Attempt budget')).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});
