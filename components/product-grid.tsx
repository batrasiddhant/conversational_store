"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"

interface Product {
  id: string
  name: string
  category: string
  price: number
  margin: number
  image: string
  description: string
  top_ingredients?: string
  tags?: string[]
}

interface ProductGridProps {
  category?: string | null
  products?: Product[]
}

export default function ProductGrid({ category, products }: ProductGridProps) {
  const [displayProducts, setDisplayProducts] = useState<Product[]>([])

  useEffect(() => {
    if (products !== undefined) {
      setDisplayProducts(products)
      return
    }

    // Real product data from the spreadsheet
    const realProducts: Product[] = [
      {
        id: "SRM001",
        name: "Radiant Renewal Serum",
        category: "Serum",
        description: "Brightens dull skin and smooths fine lines for a lit-from-within glow.",
        top_ingredients: "Ascorbic Acid (Vitamin C); Hyaluronic Acid; Niacinamide",
        tags: ["antiaging", "brightening", "hydration"],
        price: 72,
        margin: 0.45,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM002",
        name: "Dewdrop Hydration Elixir",
        category: "Serum",
        description: "Water-weight serum that drenches skin in long-lasting moisture and repairs the barrier.",
        top_ingredients: "Hyaluronic Acid; Panthenol; Vitamin F (Linoleic Acid)",
        tags: ["hydration", "barrier-repair", "dry-skin"],
        price: 58,
        margin: 0.42,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM003",
        name: "Clear Slate BHA Serum",
        category: "Serum",
        description: "Clarifies congested pores and keeps oil in check without over-drying.",
        top_ingredients: "Salicylic Acid; Zinc PCA; Green Tea Extract",
        tags: ["acne-prone", "oil-control", "pore-care"],
        price: 46,
        margin: 0.4,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM004",
        name: "Midnight Peptide Firming Serum",
        category: "Serum",
        description: "Overnight peptide complex visibly firms and nourishes mature skin.",
        top_ingredients: "Palmitoyl Tripeptide-5; Soluble Collagen; Vitamin E",
        tags: ["antiaging", "firming", "dry-skin"],
        price: 85,
        margin: 0.52,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM005",
        name: "Calm & Soothe Sensitive Serum",
        category: "Serum",
        description: "Lightweight gel serum that calms redness and strengthens sensitive skin.",
        top_ingredients: "Centella Asiatica; Allantoin; Beta-Glucan",
        tags: ["sensitive", "redness-relief", "hydration"],
        price: 49,
        margin: 0.38,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM006",
        name: "Flash Fade Spot Corrector",
        category: "Serum",
        description: "Targets dark spots and post-acne marks for an even, luminous tone.",
        top_ingredients: "Tranexamic Acid; Niacinamide; Licorice Root Extract",
        tags: ["brightening", "antiaging", "hyperpigmentation"],
        price: 79,
        margin: 0.5,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM007",
        name: "Velvet Matte Pore Serum",
        category: "Serum",
        description: "Blurs shine and minimizes pores for a soft-focus finish.",
        top_ingredients: "Niacinamide; Tremella Mushroom Extract; Willow-Bark",
        tags: ["oily-skin", "pore-refining", "matte"],
        price: 54,
        margin: 0.41,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM008",
        name: "AquaShield Barrier Booster",
        category: "Serum",
        description: "Ceramide-rich serum that reinforces the lipid barrier and locks in moisture.",
        top_ingredients: "Ceramides NP+AP+EOP; Vitamin F; Squalane",
        tags: ["hydration", "barrier-repair", "dry-skin"],
        price: 65,
        margin: 0.44,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM009",
        name: "Sunrise Retinal Serum",
        category: "Serum",
        description: "Next-gen retinaldehyde visibly smooths wrinkles with less irritation.",
        top_ingredients: "Retinaldehyde; Bakuchiol; Hyaluronic Acid",
        tags: ["antiaging", "fine-lines", "brightening"],
        price: 95,
        margin: 0.55,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "SRM010",
        name: "Glow Guard Antioxidant Serum",
        category: "Serum",
        description: "Potent antioxidant trio defends against pollution and UV stress.",
        top_ingredients: "Ferulic Acid; Ascorbic Acid (Vit C); Vitamin E",
        tags: ["antioxidant", "brightening", "antiaging"],
        price: 82,
        margin: 0.47,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "TNR001",
        name: "Rosewater Rescue Hydrating Toner",
        category: "Toner",
        description: "Alcohol-free rosewater blend that replenishes moisture and soothes tight, very dry skin.",
        top_ingredients: "Damask Rose Water; Glycerin; Aloe Vera Juice",
        tags: ["hydration", "very-dry", "sensitive", "natural"],
        price: 32,
        margin: 0.4,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "TNR002",
        name: "ClearWave 2% BHA Toner",
        category: "Toner",
        description:
          "Oil-controlling liquid exfoliant unclogs pores and keeps breakouts at bay for oily, acne-prone skin.",
        top_ingredients: "Salicylic Acid 2%; Witch Hazel; Niacinamide",
        tags: ["oily", "acne-prone", "pore-care", "BHA"],
        price: 38,
        margin: 0.45,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "TNR003",
        name: "GlowPrep 5% Glycolic Toner",
        category: "Toner",
        description: "Daily AHA toner that sweeps away dull cells and brightens uneven tone.",
        top_ingredients: "Glycolic Acid 5%; Lactic Acid; Rooibos Extract",
        tags: ["AHA", "brightening", "combo-skin"],
        price: 42,
        margin: 0.47,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "TNR004",
        name: "CalmCucumber pH Mist",
        category: "Toner",
        description: "Ultra-gentle mist balances pH and calms redness—ideal for reactive or sensitized skin.",
        top_ingredients: "Cucumber Water; Allantoin; Panthenol",
        tags: ["sensitive", "redness-relief", "natural", "hydration"],
        price: 28,
        margin: 0.36,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "TNR005",
        name: "SmoothStart PHA Tonic",
        category: "Toner",
        description:
          "Mild polyhydroxy toner refines texture without irritation—great for combination or first-time exfoliators.",
        top_ingredients: "Gluconolactone (PHA); Green Tea Extract; Hyaluronic Acid",
        tags: ["combo-skin", "gentle-exfoliation", "PHA"],
        price: 35,
        margin: 0.42,
        image: `/placeholder.svg?height=200&width=200`,
      },
      // Adding more products from other categories
      {
        id: "SUN001",
        name: "HydraCloud Daily Gel-Cream SPF 30",
        category: "Sunscreen",
        description: "Weightless water-gel that floods skin with moisture while protecting against UVA/UVB.",
        top_ingredients: "Hyaluronic Acid; Niacinamide; Zinc Oxide (SPF 30)",
        tags: ["hydration", "SPF", "dry-skin", "combo-skin"],
        price: 42,
        margin: 0.4,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "CRM002",
        name: "Velvet Night Repair Cream",
        category: "Cream / Moisturizer",
        description: "Luxurious overnight treatment that boosts collagen and repairs the skin barrier while you sleep.",
        top_ingredients: "Retinol; Peptides; Squalane",
        tags: ["night-repair", "antiaging", "dry-skin"],
        price: 68,
        margin: 0.48,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "CRM003",
        name: "Radiance Spot-Bright Eye Cream",
        category: "Cream / Moisturizer",
        description: "Brightens dark circles and smooths fine lines around delicate eye area.",
        top_ingredients: "Vitamin C; Caffeine; Niacinamide",
        tags: ["eye-care", "brightening", "antiaging", "hydration"],
        price: 55,
        margin: 0.46,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "MSK001",
        name: "HydraDeep Overnight Mask",
        category: "Face Mask",
        description: "Cushiony sleeping mask that seals in moisture and plumps skin while you rest.",
        top_ingredients: "Hyaluronic Acid; Ceramides; Blue Algae Ferment",
        tags: ["overnight", "hydration", "plumping", "dry-skin"],
        price: 58,
        margin: 0.48,
        image: `/placeholder.svg?height=200&width=200`,
      },
      {
        id: "BW001",
        name: "Revive Citrus Shower Gel",
        category: "Body Wash",
        description: "Sulphate-free body wash with energizing citrus aroma and silky lather.",
        top_ingredients: "Vitamin C; Orange Essential Oil; Aloe Vera",
        tags: ["body-wash", "energizing", "sulfate-free", "aromatic"],
        price: 25,
        margin: 0.34,
        image: `/placeholder.svg?height=200&width=200`,
      },
    ]

    if (category) {
      setDisplayProducts(realProducts.filter((p) => p.category === category))
    } else {
      setDisplayProducts(realProducts)
    }
  }, [category, products])

  if (displayProducts.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No products found</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-6">
      {displayProducts.map((product) => (
        <Card key={product.id} className="overflow-hidden">
          <div className="relative h-48 bg-gray-100">
            <Image src={product.image || "/placeholder.svg"} alt={product.name} fill className="object-cover" />
          </div>
          <CardContent className="p-4">
            <div className="flex justify-between items-start mb-2">
              <Badge variant="outline">{product.category}</Badge>
              <span className="font-bold">${product.price}</span>
            </div>
            <h3 className="font-medium text-lg">{product.name}</h3>
            <p className="text-sm text-gray-500 mt-1 line-clamp-2">{product.description}</p>
          </CardContent>
          <CardFooter className="p-4 pt-0 flex flex-wrap gap-1">
            {Array.isArray(product.tags) &&
              product.tags.map((tag, i) => (
                <Badge key={i} variant="secondary" className="text-xs">
                  {tag}
                </Badge>
              ))}
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
