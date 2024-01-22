import asyncio
from tqdm.asyncio import tqdm

async def run_tasks(tasks, rate_limit):
    results = []
    total_tasks = len(tasks)

    async def task_runner(task, pbar):
        pbar.update(1)
        result = await task
        return result

    pbar = tqdm(total=total_tasks)
    for task in tasks:
        results.append(await task_runner(task, pbar))
        await asyncio.sleep(rate_limit)
    pbar.close()

    return results