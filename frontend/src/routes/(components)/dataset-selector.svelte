<script lang="ts">
  import * as Select from "$lib/components/ui/select/index.js";
  import type { Dataset } from "$lib/types";

  interface Props {
    datasets: Dataset[];
    selectedDataset: Dataset | undefined;
  }

  let { datasets, selectedDataset = $bindable() }: Props = $props();

  const triggerContent = $derived(selectedDataset?.name ?? "Select a dataset");
</script>

<Select.Root type="single" name="favoriteFruit" onValueChange={(value) => (selectedDataset = datasets.find((d) => d.name === value))}>
  <Select.Trigger size="sm" disabled={false}>
    {triggerContent}
  </Select.Trigger>
  <Select.Content>
    {#each datasets as dataset (dataset.name)}
      <Select.Item value={dataset.name} label={dataset.name}>
        {dataset.name}
      </Select.Item>
    {/each}
  </Select.Content>
</Select.Root>
