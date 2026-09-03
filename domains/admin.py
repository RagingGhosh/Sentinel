from django.contrib import admin

from domains.models import Category, Domain


class CategoryInline(admin.TabularInline):
    model = Category
    extra = 0


@admin.register(Domain)
class DomainAdmin(admin.ModelAdmin):
    list_display = ("slug", "name", "is_active", "created_at")
    list_filter = ("is_active",)
    search_fields = ("slug", "name")
    inlines = [CategoryInline]


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("domain", "slug", "name", "sla_hours")
    list_filter = ("domain",)
    search_fields = ("slug", "name")
