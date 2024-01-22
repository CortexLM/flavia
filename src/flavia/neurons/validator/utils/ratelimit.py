import asyncio

from tqdm import tqdm
async def run_tasks(tasks, rate_limit):
    results = []
    total_tasks = len(tasks)
    semaphore = asyncio.Semaphore(rate_limit)  # Create a semaphore with the rate limit

    async def task_runner(task, progress_bar):
        async with semaphore:
            result = await asyncio.ensure_future(task)
            results.append(result)
            progress_bar.update(1)  # Update the progress bar after each task completion

    with tqdm(total=total_tasks) as pbar:
        task_futures = [task_runner(task, pbar) for task in tasks]
        await asyncio.gather(*task_futures)

    return results