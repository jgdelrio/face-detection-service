import json
from time import time
from aiohttp import web
from src.config import PORT, ALLOWED_CONTENT, DFT_MODEL
from src.tools import img2np, show_np_img
from src.face_detection import LOGGER, process_image, get_model_reference, unavailable_model


async def healthcheck(request):
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return web.Response(text=json.dumps("Face detection API is healthy!"), headers=headers, status=200)


async def image_request(request):
    try:
        # Gather input parameters and select model
        request_content_type = request.headers.get("Content-Type")
        data = await request.post()
        if 'image' not in data.keys():
            return web.Response(text=json.dumps(f"Invalid request. No image key."), status=400)
        model_info = get_model_reference(data['model']) if 'model' in data.keys() else get_model_reference(DFT_MODEL)
        model_params = data['model_params'] if 'model_params' in data.keys() else None
        mod_image = bool(data['mod_image']) if 'mod_image' in data.keys() else False
        print_label = bool(data['print_label']) if 'print_label' in data.keys() else False
    except ValueError as err:
        return web.Response(text=json.dumps(f"Error reading the request: {err}"), status=400)
    except Exception as err:
        return web.Response(text=json.dumps(f"Error on request: {err}"), status=400)

    try:
        t_ini = time()
        content_type = data['image'].content_type if hasattr(data['image'], "content_type") else None
        if content_type not in ALLOWED_CONTENT:
            return web.Response(text=json.dumps(f"Content not allowed"), status=400)
        filename = data['image'].filename
        headers = data['image'].headers
        raw_img = data['image'].file.read()
        LOGGER.info(f"Processing {filename} ...")

        np_img = img2np(raw_img)
        # show_np_img(np_img)
        results = process_image(np_img, model_info, model_params, mod_image, print_label)
        t_end = time() - t_ini
        LOGGER.info({"Processing": "finished", "file": filename, "model": model_info.name,
                               "total_time": f"{t_end:0.6}", "inference_time": f"{results['inference_time']:0.6}"})
        return web.Response(
            body=json.dumps({"file": filename, "data": results,
                             "model": model_info.name, "version": model_info.version}),
            headers=dict({"Content-Type": "application/json"}),
            status=200)

    except Exception as err:
        LOGGER.error(f"Internal server error: {err}")
        return web.Response(text=json.dumps(f"Internal server error {err}"), status=500)


"""Define app and API endpoints"""
app = web.Application()
app.router.add_get("/face-detection/healthcheck", healthcheck)
app.router.add_post("/face-detection", image_request)


if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT)
