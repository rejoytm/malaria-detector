import { getDatasets, getModels } from "$lib/api";

export const load = async () => {
  return {
    datasets: getDatasets(),
    models: getModels(),
  };
};
