import asyncio
from tqdm.asyncio import tqdm

async def run_tasks(tasks, rate_limit):
    results = []
    total_tasks = len(tasks)
    update_interval = total_tasks // 5  # Calculer l'intervalle de mise à jour pour 20%
    next_update_step = update_interval

    async def task_runner(task):
        return await task

    pbar = tqdm(total=total_tasks)
    for i, task in enumerate(tasks):
        results.append(await task_runner(task))

        # Vérifier si le prochain seuil de mise à jour a été atteint
        if (i + 1) == next_update_step or i == total_tasks - 1:
            pbar.update(next_update_step - pbar.n)  # Mettre à jour la progression de la barre
            next_update_step += update_interval  # Définir le seuil pour la prochaine mise à jour

        await asyncio.sleep(rate_limit)
    pbar.close()

    return results