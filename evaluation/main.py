import socketio
import asyncio

from namespaces import EvaluationNamespace
from config import config

sio = socketio.AsyncClient()

#TODO: Test if it works, including the cache handler
async def main():
    eval_namespace = EvaluationNamespace("/evaluation")
    sio.register_namespace(eval_namespace)

    await sio.connect(
        config["SOCKET_DOMAIN"], 
        headers={
            "serverid": config["SERVER_ID"]
        }
        # auth={ 
        #     config["AUTH_TOKEN_KEY"]: config["AUTH_TOKEN"] 
        # }
    )

    task1 = asyncio.create_task(
        handle_cache(eval_namespace)
    )
    
    task2 = asyncio.create_task(
        await sio.wait()
    )
    

async def handle_cache(eval_namespace): 
    while True:
        await asyncio.sleep(180000)
        eval_namespace.clear_unused_from_cache()


if __name__ == "__main__":
    # eval_namespace = EvaluationNamespace("/evaluation")
    asyncio.run(main())
