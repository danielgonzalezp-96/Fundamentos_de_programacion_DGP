##Método asíncrono

import asyncio
import time
import aiohttp

async def download_site(session, url):
    async with session.get(url) as response:
        data = await response.read()
        print(f"Read {len(data)} from {url}")
        
async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        tasks = [download_site(session, url) for url in sites]
        await asyncio.gather(*tasks)
        
async def main():
    sites = ["https://picsum.photos/600/400?random=1",
    "https://picsum.photos/600/400?random=2",
    "https://picsum.photos/600/400?random=3",
    "https://picsum.photos/600/400?random=4","https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800", "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/users/1","https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
    "https://raw.githubusercontent.com/python/cpython/main/README.rst",
    "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
    "https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md","https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "https://www.africau.edu/images/default/sample.pdf","https://www.soundjay.com/misc/sounds/bell-ringing-05.wav","https://raw.githubusercontent.com/django/django/main/setup.cfg",
    "https://raw.githubusercontent.com/pallets/flask/main/pyproject.toml",
    "https://raw.githubusercontent.com/numpy/numpy/main/requirements.txt",
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/setup.py"
    ]
    start_time = time.time()
    await download_all_sites(sites) #Agregué await para esperar a que se complete la tarea, con asyncio se retornaba un Runtime Error
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {time.time()-start_time:.3f} seconds")
    
if __name__ == "__main__":
    asyncio.run(main())