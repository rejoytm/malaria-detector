<script lang="ts">
  import { cn } from "$lib/utils";
  import Checkbox from "$lib/components/ui/checkbox/checkbox.svelte";
  import Separator from "$lib/components/ui/separator/separator.svelte";
  import type { Model } from "$lib/types";

  interface Props {
    models: Model[];
    selectedModels: Model[];
  }

  let { models, selectedModels = $bindable() }: Props = $props();

  let originalModels = $derived(models.filter((m) => m.category === "original"));
  let normalizedModels = $derived(models.filter((m) => m.category === "normalized"));
  let fusionModels = $derived(models.filter((m) => m.category === "fusion"));

  function toggleModelSelection(model: Model) {
    if (!selectedModels.some((m) => m.name === model.name)) {
      selectedModels.push(model);
    } else {
      selectedModels = selectedModels.filter((m) => m.name !== model.name);
    }
  }
</script>

<div class="bg-background rounded-xl shadow py-4 px-6">
  <h2 class="font-bold">Models</h2>
  <p class="text-sm mt-0.5 text-muted-foreground">
    {#if selectedModels.length}
      <span>{selectedModels.length} model{selectedModels.length > 1 ? "s" : ""} selected</span>
    {:else}
      Choose one or more models to compare
    {/if}
  </p>

  <Separator class="mt-4" />

  {@render modelCategory("Trained on Cell Images Dataset", originalModels)}
  {@render modelCategory("Trained on Stain-Normalized Images", normalizedModels)}
  {@render modelCategory("Fusion Models", fusionModels)}
</div>

{#snippet modelCategory(category: string, models: Model[])}
  <h3 class="font-semibold text-sm mt-4">{category}</h3>
  <ul class="mt-1.5">
    {#each models as model}
      {@const isModelSelected = selectedModels.some((m) => m.name === model.name)}
      <li class="text-sm -mx-3">
        <button onclick={() => toggleModelSelection(model)} class={cn("text-left flex gap-3 hover:bg-muted px-3 py-2 w-full transition rounded-md")}>
          <Checkbox checked={isModelSelected} class="mt-0.5" />
          <div>
            <p>
              {model.name}
              {#if model.is_starred}
                <span class="ml-1">⭐</span>
              {/if}
            </p>
            <p class="text-muted-foreground">{model.description}</p>
          </div>
        </button>
      </li>
    {/each}
  </ul>
{/snippet}
