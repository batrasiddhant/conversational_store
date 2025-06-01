"use client"

import { useState } from "react"
import ProductGrid from "@/components/product-grid"
import ConversationalSearch from "@/components/conversational-search"
import CategoryFilter from "@/components/category-filter"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function Home() {
  const [activeCategory, setActiveCategory] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<any[] | null>(null)

  return (
    <main className="min-h-screen bg-white">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight mb-2">Everglow - Conversational Store</h1>
          <p className="text-gray-500">Your personal skincare assistant</p>
        </header>

        <ConversationalSearch onResults={setSearchResults} />

        <Tabs defaultValue="browse" className="mt-8">
          <TabsList>
            <TabsTrigger value="browse">Browse Products</TabsTrigger>
            <TabsTrigger value="results">Search Results</TabsTrigger>
          </TabsList>

          <TabsContent value="browse">
            <CategoryFilter onCategoryChange={setActiveCategory} />
            <ProductGrid category={activeCategory} />
          </TabsContent>

          <TabsContent value="results">
            {searchResults ? (
              <div>
                <h2 className="text-xl font-semibold mb-4">Recommended Products</h2>
                <ProductGrid products={searchResults} />
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-gray-500">Use the search above to find products</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
