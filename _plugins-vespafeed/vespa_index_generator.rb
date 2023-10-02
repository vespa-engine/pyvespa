# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

require 'json'
require 'nokogiri'
require 'kramdown/parser/kramdown'

module Jekyll

    class VespaIndexGenerator < Jekyll::Generator
        priority :lowest

        def generate(site)
            namespace = site.config["search"]["namespace"]
            operations = []
            site.pages.each do |page|
                next if (page.path.start_with?("search.html") ||
                         page.path.start_with?("genindex.html") ||
                         page.url.start_with?("/redirects.json"))
                if page.data["index"] == true
                    title = clean_chars(extract_title(page))
                    text  = clean_chars(extract_text(page))
                    operations.push({
                        :put => "id:"+namespace+":doc::"+namespace+page.url,
                        :fields => {
                            :path => page.url,
                            :namespace => namespace,
                            :title => title,
                            :content => text,
                            :html => get_html(page),
                            :term_count => text.split.length(),
                            :last_updated => Time.now.to_i,
                            :outlinks => extract_links(page)
                        }
                    })
                end
            end

            json = JSON.pretty_generate(operations)
            File.open(namespace + "_index.json", "w") { |f| f.write(json) }
        end

        def strip_a_from_headers(htmldoc)
            doc = Nokogiri::HTML(htmldoc)
            doc.css('h1 a').each { |e| e.remove }
            doc.css('h2 a').each { |e| e.remove }
            doc.css('h3 a').each { |e| e.remove }
            doc.css('h4 a').each { |e| e.remove }
            doc.css('h5 a').each { |e| e.remove }
            return doc.to_html
        end

        def get_html(page)
            # all pyvespa is on HTML pages (for now)
            strip_a_from_headers(page.content)
        end

        def extract_text(page)
            ext = page.name[page.name.rindex('.')+1..-1]
            if ext == "md"
                input = Kramdown::Document.new(page.content).to_html
            else
                input = page.content
            end
            doc = Nokogiri::HTML(input)
            doc.search('th,td').each{ |e| e.after "\n" }
            doc.search('style').each{ |e| e.remove }
            content = doc.xpath("//text()").to_s
            page_text = content.gsub("\r"," ").gsub("\n"," ")
        end

        def extract_links(page)
            doc = Nokogiri::HTML(page.content)
            links = doc.css('a').map { |link| link['href']}
        end

        def extract_title(page)
            doc = Nokogiri::HTML(page.content)
            headings = doc.xpath('//h1')
            if headings.length > 0
                title = headings[0].text
            else
                title = "pyvespa"
            end
        end

        def clean_chars(string)
          string.gsub('', '')
        end

    end

end
