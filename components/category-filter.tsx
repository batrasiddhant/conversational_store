"use client"

import { Button } from "@/components/ui/button"

interface CategoryFilterProps {
  onCategoryChange: (category: string | null) => void
}

export default function CategoryFilter({ onCategoryChange }: CategoryFilterProps) {
  // Updated categories based on the spreadsheet data
  const categories = [
    "All",
    "Serum",
    "Toner",
    "Sunscreen",
    "Cream / Moisturizer",
    "Face Mask",
    "Body Wash",
    "Shampoo",
    "Conditioner",
    "Hair Mask",
  ]

  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {categories.map((category) => (
        <Button key={category} variant="outline" onClick={() => onCategoryChange(category === "All" ? null : category)}>
          {category}
        </Button>
      ))}
    </div>
  )
}
