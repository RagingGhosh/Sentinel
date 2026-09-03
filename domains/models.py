from django.db import models


class Domain(models.Model):
    slug = models.SlugField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        permissions = [("manage_domain", "Can manage domains and categories")]

    def __str__(self) -> str:
        return self.name


class Category(models.Model):
    domain = models.ForeignKey(Domain, on_delete=models.CASCADE, related_name="categories")
    slug = models.SlugField(max_length=100)
    name = models.CharField(max_length=200)
    sla_hours = models.PositiveIntegerField(
        help_text="Hours from human triage confirmation until the complaint is due."
    )

    class Meta:
        ordering = ["domain", "name"]
        constraints = [
            models.UniqueConstraint(
                fields=["domain", "slug"], name="unique_category_slug_per_domain"
            ),
            models.CheckConstraint(condition=models.Q(sla_hours__gt=0), name="sla_hours_positive"),
        ]

    def __str__(self) -> str:
        return f"{self.domain.slug}/{self.slug}"
