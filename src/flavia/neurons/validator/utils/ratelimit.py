import asyncio
from tqdm.asyncio import tqdm

async def run_tasks(tasks, rate_limit):
    results = []
    total_tasks = len(tasks)
    update_interval = total_tasks // 5

    async def task_runner(task):
        return await task

    pbar = tqdm(total=total_tasks)
    for i, task in enumerate(tasks):
        results.append(await task_runner(task))
        if (i + 1) % update_interval == 0 or i == total_tasks - 1:
            pbar.update(update_interval if (i + 1) % update_interval == 0 else total_tasks % update_interval)
        await asyncio.sleep(rate_limit)
    pbar.close()

    return results