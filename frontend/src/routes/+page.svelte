<script lang="ts">
  import { onMount } from "svelte";
  import { CircleAlert } from "@lucide/svelte";
  import * as Empty from "$lib/components/ui/empty/index.js";
  import Spinner from "$lib/components/ui/spinner/spinner.svelte";
  import ModelSelector from "./(components)/model-selector.svelte";
  import TestsSection from "./(components)/tests-section.svelte";

  import type { DataBucket, Dataset, Model, ProcessingResult, Sample } from "$lib/types.js";

  let { data } = $props();

  let bucket: DataBucket<{
    datasets: Dataset[];
    models: Model[];
  }> = $state({
    isLoading: true,
    data: undefined,
    error: undefined,
  });

  let selectedDataset: Dataset | undefined = $state();
  let selectedModels: Model[] = $state([]);
  let samples: Sample[] = $state([]);
  let processingResults: ProcessingResult[] | undefined = $state();
  let selectedProcessingResult: ProcessingResult | undefined = $state();

  onMount(async () => {
    const [datasetsResponse, modelsResponse] = await Promise.all([data.datasets.catch(() => undefined), data.models.catch(() => undefined)]);

    const setError = (error: Error, occuredWhile?: string) => {
      bucket = {
        isLoading: false,
        data: undefined,
        error: {
          name: "DataAwaitError",
          message: occuredWhile ? `An error occurred while ${occuredWhile}.` : error.message,
        },
      };
    };

    if (!datasetsResponse || !modelsResponse) {
      const context = !datasetsResponse ? "awaiting datasets" : !modelsResponse ? "awaiting models" : "awaiting data";
      setError(new Error(), context);
      return;
    }

    if (datasetsResponse.error) {
      setError(datasetsResponse.error);
      return;
    }

    if (modelsResponse.error) {
      setError(modelsResponse.error);
      return;
    }

    bucket = {
      isLoading: false,
      data: {
        datasets: datasetsResponse.data,
        models: modelsResponse.data,
      },
      error: undefined,
    };

    selectedDataset = bucket.data.datasets[0];
  });
</script>

<main class="bg-muted min-h-dvh">
  <section class="p-6 w-full max-w-7xl mx-auto">
    {#if bucket.isLoading}
      <div class="min-h-[calc(100dvh-(--spacing(6))*2)] flex justify-center items-center text-sm">
        <Spinner class="mr-2" /> Loading...
      </div>
    {:else if bucket.error}
      <div class="min-h-[calc(100dvh-(--spacing(6))*2)] flex justify-center items-center">
        <Empty.Root>
          <Empty.Header>
            <Empty.Media variant="icon">
              <CircleAlert />
            </Empty.Media>
            <Empty.Title>An error occured</Empty.Title>
            <Empty.Description>{bucket.error.name}: {bucket.error.message}</Empty.Description>
          </Empty.Header>
        </Empty.Root>
      </div>
    {:else}
      <div class="grid grid-cols-[--spacing(96)_1fr] gap-8">
        <ModelSelector models={bucket.data.models} bind:selectedModels />
        <TestsSection datasets={bucket.data.datasets} bind:selectedDataset {selectedModels} bind:samples {processingResults} bind:selectedProcessingResult />
      </div>
    {/if}
  </section>
</main>
